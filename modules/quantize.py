from typing import NamedTuple, Tuple

from modules.loss import QuantizeLoss
from modules.utils import kmeans

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import torch.distributed as dist


def efficient_rotation_trick_transform(u: Tensor, q: Tensor):
    """
    4.2 in https://arxiv.org/abs/2410.06424
    args:
        u: [B, D]
        q: [B, D]
    returns:
        transformed: [B, D]
    """
    u_norm = torch.norm(u, dim=-1, keepdim=True).clamp(min=1e-6)
    q_norm = torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-6)
    u_bar = u / u_norm
    q_bar = q / q_norm
    w = F.normalize(u_bar + q_bar, p=2, dim=1, eps=1e-6)

    u = u.unsqueeze(1) # [B, 1, D]
    return (
        u -
        2 * (u @ w.unsqueeze(-1).detach() @ w.unsqueeze(1).detach()) +
        2 * (u @ u_bar.unsqueeze(-1).detach() @ q_bar.unsqueeze(1).detach())
    ).squeeze(1) * (q_norm / u_norm).detach()

def sinkhorn(cost: Tensor, n_iters: int, epsilon: float = 10) -> Tensor:
    """
    3.1 in https://arxiv.org/abs/2006.09882
    """
    B, N = cost.shape
    Q = torch.exp(-cost * epsilon).t()
    sum_Q = Q.sum()
    is_distributed = dist.is_available() and dist.is_initialized()
    if is_distributed:
        B = dist.get_world_size() * B
        dist.all_reduce(sum_Q)
    Q /= (sum_Q + 1e-8)
    for _ in range(n_iters):
        sum_of_rows = Q.sum(dim=1, keepdim=True)
        if is_distributed:
            dist.all_reduce(sum_of_rows)
        Q /= (sum_of_rows.clamp(min=1e-8))
        Q /= N

        Q /= (Q.sum(dim=0, keepdim=True).clamp(min=1e-8))
        Q /= B
    Q *= B
    return Q.t()

def sample_gumble(shape: Tuple, device: torch.device, eps=1e-20):
    """sample from Gumbel(0, 1)"""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits: Tensor, temperature: float = 0.2):
    y = logits + sample_gumble(logits.shape, logits.device)
    return F.softmax(y / temperature, dim=-1)

class QuantizeOutput(NamedTuple):
    embeddings: Tensor
    ids: Tensor
    loss: Tensor
    score: Tensor

class Quantize(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_embed: int,
        do_kmeans_init: bool = True,
        commitment_weight: float = 0.25,

        forward_mode: str = 'ste', # ['rotation_trick', 'gumbel_softmax','ste']
        gumbel_temperature: float = 0.2,

        sim_vq: bool = False,
        freeze_embedding: bool = False,

        ema: bool = False,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        ema_dead_threshold: int = 3,

        use_sinkhorn: bool = False,
        sinkhorn_iters: int = 5,
        sinkhorn_epsilon: float = 10.0,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.embeddings = nn.Embedding(n_embed, embed_dim)
        nn.init.uniform_(self.embeddings.weight)
        self.do_kmeans_init = do_kmeans_init
        self.register_buffer('kmeans_initted', torch.tensor(False))

        # forward mode
        self.forward_mode = forward_mode
        self.gumbel_temperature = gumbel_temperature

        # sinkhorn
        self.use_sinkhorn = use_sinkhorn
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_epsilon = sinkhorn_epsilon

        if ema:
            if sim_vq:
                raise ValueError("EMA is not compatible with SIM-VQ.")
            if forward_mode == 'gumbel_softmax':
                raise ValueError("EMA is not compatible with gumbel_softmax forward mode.")
        
        # simvq:  https://arxiv.org/pdf/2411.02038
        self.sim_vq = sim_vq
        self.freeze_embedding = freeze_embedding
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False) if sim_vq else nn.Identity()
        if self.freeze_embedding:
            # In simvq, freeze codebook entries so only projection (when enabled) is updated.
            self.embeddings.weight.requires_grad_(False)
        
        # ema
        self.ema = ema
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.ema_dead_threshold = ema_dead_threshold
        if self.ema:
            self.register_buffer('ema_cluster_size', torch.zeros(n_embed))
            self.register_buffer('ema_embed_avg', self.embeddings.weight.data.clone())
            self.embeddings.weight.requires_grad_(False)
        
        self.quantize_loss = QuantizeLoss(commitment_weight)

    @property
    def weight(self) -> Tensor:
        return self.embeddings.weight

    @property
    def device(self) -> torch.device:
        return self.embeddings.weight.device
    
    @torch.no_grad()
    def _ema_update(self, x: Tensor,  ids: Tensor) -> None:
        # x: [B, D], ids: [B]
        one_hot = F.one_hot(ids, num_classes=self.n_embed).type_as(x)   # [B, n_embed]
        batch_cluster_size = one_hot.sum(dim=0)                         # [n_embed]
        batch_embed_sum = one_hot.t() @ x                               # [n_embed, D]
        distributed = dist.is_available() and dist.is_initialized()
        if distributed:
            dist.all_reduce(batch_cluster_size)
            dist.all_reduce(batch_embed_sum)
        self.ema_cluster_size.mul_(self.ema_decay).add_(batch_cluster_size, alpha=1 - self.ema_decay)
        self.ema_embed_avg.mul_(self.ema_decay).add_(batch_embed_sum, alpha=1 - self.ema_decay)

        dead_mask = self.ema_cluster_size < self.ema_dead_threshold
        if dead_mask.any():
            num_dead = dead_mask.sum().item()
            new_points = x[torch.randint(0, x.size(0), (num_dead,), device=x.device)]
            if distributed:
                dist.broadcast(new_points, src=0)
            self.ema_embed_avg[dead_mask] = new_points
            self.ema_cluster_size[dead_mask] = self.ema_dead_threshold
        n = self.ema_cluster_size.sum()
        cluster_size = (self.ema_cluster_size + self.ema_eps) / (n + self.n_embed * self.ema_eps) * n
        new_embed = self.ema_embed_avg / cluster_size.unsqueeze(-1).clamp(min=self.ema_eps)
            
        self.embeddings.weight.data.copy_(new_embed)

    
    def forward(self, x: Tensor) -> QuantizeOutput:
        assert x.shape[-1] == self.embed_dim

        if self.do_kmeans_init and not self.kmeans_initted:
            centroids, _ = kmeans(x, self.n_embed)
            self.embeddings.weight.data.copy_(centroids)
            if self.ema:
                self.ema_embed_avg.data.copy_(centroids)
                self.ema_cluster_size.data.fill_(1.0)
            self.kmeans_initted.fill_(True)
        
        codebook = self.out_proj(self.embeddings.weight)
           
        dist = (
            x.pow(2).sum(dim=-1, keepdim=True)
            - 2 * x @ codebook.T
            + codebook.pow(2).sum(dim=-1, keepdim=True).T
        )

        if self.training and self.use_sinkhorn:
            transport_plan = sinkhorn(dist.detach(), self.sinkhorn_iters, self.sinkhorn_epsilon)
            ids = torch.argmax(transport_plan, dim=-1)
        else:
            ids = torch.argmin(dist.detach(), dim=-1)

        emb = self.embeddings(ids)

        if self.training:
            # EMA update if use ema
            if self.ema:
                self._ema_update(x.detach(), ids)

            if self.forward_mode == 'rotation_trick':
                emb = self.out_proj(emb)
                emb_out = efficient_rotation_trick_transform(x, emb)
            elif self.forward_mode == 'ste':
                emb = self.out_proj(emb)
                emb_out = x + (emb - x).detach()
            elif self.forward_mode == 'gumbel_softmax':
                logits = -dist
                soft_assign = gumbel_softmax_sample(logits, temperature=self.gumbel_temperature)
                emb = soft_assign @ codebook
                emb_out = emb
            else:
                raise ValueError(f"Unsupported forward mode: {self.forward_mode}")
            loss = self.quantize_loss(x, emb)
        else:
            emb_out = self.out_proj(emb)
            loss = self.quantize_loss(x, emb_out)

        return QuantizeOutput(emb_out, ids, loss, -dist)
    
