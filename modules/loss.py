from typing import List, Tuple
from dataclasses import dataclass, field, asdict

from modules.utils import GatherWithGrad

import torch
from torch import nn
from torch import Tensor
import torch.distributed as dist
import torch.nn.functional as F

class ReconstructionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_hat: Tensor, x: Tensor) -> Tensor:
        return ((x_hat - x)**2).sum(axis=-1)

class QuantizeLoss(nn.Module):
    def __init__(self, commitment_weight: float = 1.0) -> None:
        super().__init__()
        self.commitment_weight = commitment_weight

    def forward(self, query: Tensor, value: Tensor) -> Tensor:
        emb_loss = ((query.detach() - value)**2).sum(axis=[-1])
        query_loss = ((query - value.detach())**2).sum(axis=[-1])
        return emb_loss + self.commitment_weight * query_loss

class AuxiliaryLoss(nn.Module):
    def __init__(
        self,
        act_func: str = 'sigmoid', # ['sigmoid', 'softmax']
        layer_weights: List[str] = [1, 1, 1]
    ) -> None:
        super().__init__()
        self.act_func = act_func
        self.layer_weights = layer_weights

    def forward(self, x: Tensor) -> Tensor:
        """
        Auxiliary loss
        input:
            x: [B, ..., N] the score matrix
        """
        if self.act_func == 'sigmoid':
            scores = torch.sigmoid(x)
            scores = scores / scores.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        elif self.act_func == 'softmax':
            scores = F.softmax(x, dim=-1)
        else:
            raise ValueError(f'Unsupported act_func: {self.act_func}')
        if dist.is_available() and dist.is_initialized():
            all_scores = torch.cat(GatherWithGrad.apply(scores), dim=0)
        else:
            all_scores = scores
        mean_prob = all_scores.mean(dim=0) + 1e-6
        return (
            (mean_prob * torch.log(mean_prob))
            * (torch.tensor(self.layer_weights, dtype=x.dtype, device=x.device)).unsqueeze(-1)
        ).sum()

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z: Tensor) -> Tensor:
        B = z.shape[0] // 2
        z_a = F.normalize(z[:B], dim=-1)
        z_b = F.normalize(z[B:], dim=-1)

        labels = torch.arange(B, device=z.device)
        temperature = max(float(self.temperature), 1e-6)

        logits_ab = z_a @ z_b.T / temperature
        loss_ab = F.cross_entropy(logits_ab, labels)

        logits_ba = z_b @ z_a.T / temperature
        loss_ba = F.cross_entropy(logits_ba, labels)

        return (loss_ab + loss_ba) / 2

class PrefixInfoNceLoss(nn.Module):
    """Pair-based prefix InfoNCE loss.
    Input embs [2B, n_layers, embed_dim] where (i, i+B) form positive pairs.
    Computes per-layer InfoNCE on the first n_layers-1 layers.
    """
    def __init__(
        self,
        temperature: float = 0.1,
        layer_weights: List[float] = [1, 0.1, 0.01],
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.cl_loss = ContrastiveLoss(temperature=temperature)
        self.layer_weights = layer_weights

    def forward(self, embs: Tensor) -> Tensor:
        """
        embs: [2B, n_layers, embed_dim]
        returns: scalar loss
        """
        num_layers = embs.shape[1]
        if num_layers <= 1:
            return torch.tensor(0.0, device=embs.device, dtype=embs.dtype)

        loss = torch.tensor(0.0, device=embs.device, dtype=embs.dtype)
        denom_layers = num_layers - 1

        for d in range(denom_layers):
            wd = self.layer_weights[d] if d < len(self.layer_weights) else 1.0
            layer_loss = self.cl_loss(embs[:, d, :])
            loss = loss + float(wd) * layer_loss

        return loss / denom_layers


class PrefixCollisionLoss(nn.Module):
    """Prefix-aware collision loss.
    Uses common prefix length to detect collisions.
    Per-depth margins and weights control repulsion strength.
    Pushes apart colliding items in encoder representation space.
    """
    def __init__(
        self,
        prefix_margins: List[float] = [0.6, 0.8, 0.9],
        prefix_weights: List[float] = [0.1, 0.5, 1.0],
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.prefix_margins = prefix_margins
        self.prefix_weights = prefix_weights
        self.eps = eps

    def _common_prefix_length(self, sids: Tensor) -> Tensor:
        B, L = sids.shape
        match = torch.ones(B, B, dtype=torch.bool, device=sids.device)
        prefix_len = torch.zeros(B, B, dtype=torch.long, device=sids.device)
        for l in range(L):
            match = match & (sids[:, l].unsqueeze(0) == sids[:, l].unsqueeze(1))
            prefix_len += match.long()
        return prefix_len

    def forward(self, z: Tensor, sids: Tensor, mask: Tensor) -> Tensor:
        """
        z: [B, D] encoder output
        sids: [B, L] semantic IDs
        mask: [B, B] bool, True for valid pairs
        """
        B, L = sids.shape
        z_norm = F.normalize(z, dim=-1)
        D = z_norm @ z_norm.T

        # Common prefix length
        prefix_len = self._common_prefix_length(sids)  # [B, B]

        loss = torch.tensor(0.0, device=z.device, dtype=z.dtype)

        for k in range(L):
            depth = k + 1
            margin = self.prefix_margins[k] if k < len(self.prefix_margins) else self.prefix_margins[-1]
            weight = self.prefix_weights[k] if k < len(self.prefix_weights) else self.prefix_weights[-1]

            # Collision at this depth: prefix >= depth AND valid pair
            collision_mask = mask & (prefix_len >= depth)

            if not collision_mask.any():
                continue

            depth_loss = torch.relu(margin - D)
            collision_count = collision_mask.sum().clamp(min=self.eps)
            depth_loss = (depth_loss * collision_mask).sum() / collision_count

            loss = loss + weight * depth_loss

        return loss


# ======================== Loss Config Dataclasses ========================

@dataclass
class AuxiliaryLossConfig:
    act_func: str = 'sigmoid'
    layer_weights: List[float] = field(default_factory=lambda: [2, 1, 1])
    loss_weight: float = 0.01

@dataclass
class GlobalQuantizeLossConfig:
    commitment_weight: float = 0.1
    loss_weight: float = 0.1

@dataclass
class InfoNceConfig:
    temperature: float = 0.1
    layer_weights: List[float] = field(default_factory=lambda: [1, 0.1, 0.01])
    loss_weight: float = 0.01

@dataclass
class EncoderClConfig:
    temperature: float = 0.1
    loss_weight: float = 0.1

@dataclass
class PrefixCollisionConfig:
    prefix_margins: List[float] = field(default_factory=lambda: [0.6, 0.8, 0.9])
    prefix_weights: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0])
    loss_weight: float = 1.0


# ======================== Utilities ========================

def build_pair_mask(N: int, device: torch.device) -> Tensor:
    """Build mask M [2B, 2B] for pair-structured batch.
    M[i,j] = 0 if i == j (self) or |i-j| == B (co-occurring pair)
    M[i,j] = 1 otherwise
    """
    B = N // 2
    mask = torch.ones(N, N, dtype=torch.bool, device=device)
    mask.fill_diagonal_(False)
    idx = torch.arange(B, device=device)
    mask[idx, idx + B] = False
    mask[idx + B, idx] = False
    return mask


# ======================== Criterion ========================

class RqVaeCriterion:
    """Standalone criterion for RqVae training.
    Not an nn.Module — does not pollute model state_dict.
    """
    def __init__(
        self,
        auxiliary: AuxiliaryLossConfig = None,
        global_quantize: GlobalQuantizeLossConfig = None,
        info_nce: InfoNceConfig = None,
        encoder_cl: EncoderClConfig = None,
        prefix_collision: PrefixCollisionConfig = None,
    ):
        auxiliary = auxiliary or AuxiliaryLossConfig()
        global_quantize = global_quantize or GlobalQuantizeLossConfig()
        info_nce = info_nce or InfoNceConfig()
        encoder_cl = encoder_cl or EncoderClConfig()
        prefix_collision = prefix_collision or PrefixCollisionConfig()

        # loss functions
        self.reconstruction_loss_fn = ReconstructionLoss()
        self.global_quantize_loss_fn = QuantizeLoss(global_quantize.commitment_weight)
        self.auxiliary_loss_fn = AuxiliaryLoss(
            act_func=auxiliary.act_func,
            layer_weights=auxiliary.layer_weights,
        )
        self.prefix_info_nce_loss_fn = PrefixInfoNceLoss(
            temperature=info_nce.temperature,
            layer_weights=info_nce.layer_weights,
        )
        self.encoder_cl_loss_fn = ContrastiveLoss(
            temperature=encoder_cl.temperature,
        )
        self.prefix_collision_loss_fn = PrefixCollisionLoss(
            prefix_margins=prefix_collision.prefix_margins,
            prefix_weights=prefix_collision.prefix_weights,
        )

        # loss weights
        self.global_quantize_loss_weight = global_quantize.loss_weight
        self.auxiliary_loss_weight = auxiliary.loss_weight
        self.prefix_info_nce_loss_weight = info_nce.loss_weight
        self.encoder_cl_loss_weight = encoder_cl.loss_weight
        self.prefix_collision_loss_weight = prefix_collision.loss_weight

        self._config = {
            'auxiliary': asdict(auxiliary),
            'global_quantize': asdict(global_quantize),
            'info_nce': asdict(info_nce),
            'encoder_cl': asdict(encoder_cl),
            'prefix_collision': asdict(prefix_collision),
        }

    @property
    def config(self) -> dict:
        import copy
        return copy.deepcopy(self._config)

    @classmethod
    def from_config(cls, config: dict) -> 'RqVaeCriterion':
        return cls(
            auxiliary=AuxiliaryLossConfig(**config['auxiliary']),
            global_quantize=GlobalQuantizeLossConfig(**config['global_quantize']),
            info_nce=InfoNceConfig(**config['info_nce']),
            encoder_cl=EncoderClConfig(**config['encoder_cl']),
            prefix_collision=PrefixCollisionConfig(**config['prefix_collision']),
        )

    def __call__(self, output, x: Tensor, use_contrastive: bool = False) -> dict:
        """
        output: RqVaeOutput from model.forward()
        x: [B, input_dim] original input
        use_contrastive: whether to compute contrastive losses (InfoNCE, encoder CL)
        returns: loss_dict with all loss terms and total 'loss'
        """
        _zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        total_loss = _zero

        embeddings_sum = output.embeddings.sum(dim=1)

        reconstruction_loss = self.reconstruction_loss_fn(output.reconstruction, x).mean()
        quantize_loss = output.quantize_loss.sum(dim=1).mean()
        global_quantize_loss = self.global_quantize_loss_fn(query=output.z, value=embeddings_sum).mean()
        auxiliary_loss = self.auxiliary_loss_fn(output.scores)

        # Prefix collision loss (used with contrastive pair data)
        prefix_collision_loss = _zero

        total_loss = total_loss + (
            reconstruction_loss
            + quantize_loss
            + (self.global_quantize_loss_weight * global_quantize_loss if self.global_quantize_loss_weight > 0 else 0)
            + (self.auxiliary_loss_weight * auxiliary_loss if self.auxiliary_loss_weight > 0 else 0)
        )

        prefix_info_nce_loss = encoder_cl_loss = _zero

        if use_contrastive:
            prefix_info_nce_loss = self.prefix_info_nce_loss_fn(output.embeddings)
            encoder_cl_loss = self.encoder_cl_loss_fn(output.z)
            mask = build_pair_mask(x.shape[0], x.device)
            prefix_collision_loss = self.prefix_collision_loss_fn(output.z, output.sids, mask)
            total_loss = total_loss + (
                (self.prefix_info_nce_loss_weight * prefix_info_nce_loss if self.prefix_info_nce_loss_weight > 0 else 0)
                + (self.encoder_cl_loss_weight * encoder_cl_loss if self.encoder_cl_loss_weight > 0 else 0)
                + (self.prefix_collision_loss_weight * prefix_collision_loss if self.prefix_collision_loss_weight > 0 else 0)
            )

        return {
            'loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'quantize_loss': quantize_loss,
            'global_quantize_loss': global_quantize_loss,
            'auxiliary_loss': auxiliary_loss,
            'prefix_collision_loss': prefix_collision_loss,
            'prefix_info_nce_loss': prefix_info_nce_loss,
            'encoder_cl_loss': encoder_cl_loss,
        }
