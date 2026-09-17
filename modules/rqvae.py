import copy
from typing import List, Tuple
from dataclasses import dataclass, field, asdict

from modules.encoder import MLP
from modules.quantize import Quantize

import torch
from torch import nn
from torch import Tensor


torch.set_float32_matmul_precision('high')


# ======================== Config Dataclass ========================

@dataclass
class RqVaeConfig:
    input_dim: int = 128
    embed_dim: int = 64
    hidden_dims: List[int] = field(default_factory=lambda: [96])
    codebook_size: int = 256
    codebook_kmeans_init: bool = True
    quantize_layers: int = 3
    commitment_weight: float = 0.1
    forward_mode: str = 'ste'          # ['rotation_trick', 'gumbel_softmax', 'ste']
    gumbel_temperature: float = 0.2
    sim_vq: bool = False               # https://arxiv.org/pdf/2411.02038
    freeze_embedding: bool = False
    use_sinkhorn: bool = False
    sinkhorn_iters: int = 5
    sinkhorn_epsilon: float = 10.0
    ema: bool = False
    ema_dead_threshold: int = 3
    ema_decay: float = 0.99
    ema_eps: float = 1e-5
    normalize: bool = False
    dropout: float = 0.0


# ======================== Output Dataclass ========================

@dataclass
class RqVaeOutput:
    """
    z: [B, embed_dim] encoder output
    reconstruction: [B, input_dim] decoder output
    embeddings: [B, n_layers, embed_dim] per-layer quantized embeddings
    quantize_loss: [B, n_layers] per-layer quantize loss from Quantize module
    sids: [B, n_layers] semantic IDs
    scores: [B, n_layers, codebook_size] logits for each codebook
    """
    z: Tensor
    reconstruction: Tensor
    embeddings: Tensor
    quantize_loss: Tensor
    sids: Tensor
    scores: Tensor


# ======================== Model ========================

class RqVae(nn.Module):
    def __init__(self, config: RqVaeConfig = None):
        super().__init__()
        config = config or RqVaeConfig()

        self._config = asdict(config)

        # ---- quantize layers ----
        self.quantize_layers = nn.ModuleList([
            Quantize(
                embed_dim=config.embed_dim,
                n_embed=config.codebook_size,
                do_kmeans_init=config.codebook_kmeans_init,
                commitment_weight=config.commitment_weight,
                forward_mode=config.forward_mode,
                gumbel_temperature=config.gumbel_temperature,
                sim_vq=config.sim_vq,
                freeze_embedding=config.freeze_embedding,
                use_sinkhorn=config.use_sinkhorn,
                sinkhorn_iters=config.sinkhorn_iters,
                sinkhorn_epsilon=config.sinkhorn_epsilon,
                ema=config.ema,
                ema_decay=config.ema_decay,
                ema_eps=config.ema_eps,
                ema_dead_threshold=config.ema_dead_threshold,
            )
            for _ in range(config.quantize_layers)
        ])

        # ---- encoder / decoder ----
        self.encoder = MLP(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            out_dim=config.embed_dim,
            dropout=config.dropout,
            normalize=config.normalize,
        )
        self.decoder = MLP(
            input_dim=config.embed_dim,
            hidden_dims=config.hidden_dims[::-1],
            out_dim=config.input_dim,
            dropout=config.dropout,
            normalize=config.normalize,
        )

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    @property
    def config(self) -> dict:
        return copy.deepcopy(self._config)

    @classmethod
    def from_config(cls, config: dict) -> 'RqVae':
        return cls(config=RqVaeConfig(**config))
        

    def forward(self, x: Tensor) -> RqVaeOutput:
        z = self.encoder(x)
        residual = z
        embs, scores, sem_ids, quantize_loss = [], [], [], []
        for layer in self.quantize_layers:
            quantize_output = layer(residual)
            emb, ids, loss, score = (
                quantize_output.embeddings, quantize_output.ids,
                quantize_output.loss, quantize_output.score,
            )
            residual = residual - emb
            embs.append(emb)
            scores.append(score)
            sem_ids.append(ids)
            quantize_loss.append(loss)
        embs = torch.stack(embs, dim=1)
        scores = torch.stack(scores, dim=1)
        quantize_loss = torch.stack(quantize_loss, dim=1)
        sids = torch.stack(sem_ids, dim=1)
        reconstruction = self.decoder(embs.sum(dim=1))
        return RqVaeOutput(
            z=z,
            reconstruction=reconstruction,
            embeddings=embs,
            quantize_loss=quantize_loss,
            sids=sids,
            scores=scores,
        )
