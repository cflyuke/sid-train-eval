import os
import time
import copy
from dataclasses import dataclass

from modules.utils import kmeans, batch_to

from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from datasets import Dataset


class Kmeans(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_clusters: int, 
        max_iters: int = 1000, 
        tol: float = 1e-6,
        reset_threshold: int = 3,
        balance_factor: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_clusters = num_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.reset_threshold = reset_threshold
        self.balance_factor = balance_factor
        self.register_buffer('is_inited', torch.tensor(False))
        self.register_buffer('centroids', torch.zeros(num_clusters, dim))
        self.register_buffer('acc_cluster_size', torch.zeros(num_clusters))
        self.register_buffer('acc_cluster_sum', torch.zeros(num_clusters, dim))


    @torch.no_grad()
    def update_centroids(self, dataset: Dataset):
        update_mask = self.acc_cluster_size > self.reset_threshold
        dead_mask = torch.logical_not(update_mask)
        new_centorids = torch.zeros_like(self.centroids, device=self.centroids.device)
        if dead_mask.any():
            num_empty = dead_mask.sum().item()
            init_data = dataset.shuffle(seed=int(time.time())).take(int(min(num_empty * torch.max(self.acc_cluster_size).item(), 20000)))
            init_data = batch_to(init_data, self.centroids.device)
            new_points, _ = kmeans(init_data, num_empty)
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(new_points, 0)
            new_centorids[dead_mask] = new_points
        acc_cluster_size = torch.where(update_mask, self.acc_cluster_size, 1.0)
        acc_cluster_mean = torch.where(update_mask.unsqueeze(1), self.acc_cluster_sum, new_centorids) / acc_cluster_size.unsqueeze(1)

        update_norm = torch.nn.functional.pairwise_distance(self.centroids, acc_cluster_mean)
        self.centroids.copy_(acc_cluster_mean)
        self.acc_cluster_sum.zero_()
        self.acc_cluster_size.zero_()
        return update_norm

    
    @torch.no_grad()
    def forward(self, X: torch.Tensor, acc=False):
        if not self.is_inited:
            centroids, _ = kmeans(X, num_clusters=self.num_clusters)
            self.centroids.copy_(centroids)
            self.is_inited.fill_(True)
        distances = torch.cdist(X, self.centroids)
        if acc:
            total_size = self.acc_cluster_size.sum().item()
            avg_size = total_size / self.num_clusters
            if avg_size > 0:
                balance_penalty = self.balance_factor * torch.relu(self.acc_cluster_size / avg_size - 1)
                distances += balance_penalty.unsqueeze(0)
            indices = torch.argmin(distances, dim=1)
            # [num_clusters]
            tmp_cluster_size = torch.bincount(indices, minlength=self.num_clusters)
            # [num_clusters, dim]
            tmp_cluster_emb = torch.index_add(torch.zeros_like(self.acc_cluster_sum, dtype=X.dtype, device=X.device), 0, indices, X)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(tmp_cluster_size)
                dist.all_reduce(tmp_cluster_emb)
            
            self.acc_cluster_size += tmp_cluster_size
            self.acc_cluster_sum += tmp_cluster_emb
        else:
            indices = torch.argmin(distances, dim=1)
        return self.centroids[indices], indices

    @torch.no_grad()
    def fit(self, data: DataLoader):
        is_main_process = os.environ.get('RANK', '0') == '0'
        pbar = tqdm(range(self.max_iters), desc='K-means', disable=not is_main_process, position=0)
        for _ in pbar:
            for batch in data:
                batch = batch['embedding']
                self.forward(batch, acc=True)
            shift = self.update_centroids(data.dataset).max()
            pbar.set_postfix(shift=f'{shift.item():.6f}')
            if shift < self.tol:
                break
        pbar.close()


@dataclass
class RqKmeansOutput:
    sids: torch.Tensor
    loss_dict: dict

class RqKmeans(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        n_layers: int = 3,
        max_iters: int = 1000,
        tol: float = 1e-6,
        reset_threshold: int = 3,
        balance_factor: float = 0.1,
    ):
        cfg = dict(locals())
        cfg.pop('self', None)
        cfg.pop('__class__', None)
        self._config = copy.deepcopy(cfg)
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.n_layers = n_layers
        self.tol = tol
        self.max_iters = max_iters
        self.reset_threshold = reset_threshold
        self.balance_factor = balance_factor

        self.layers = nn.ModuleList([Kmeans(dim, codebook_size, max_iters, tol, reset_threshold, balance_factor) for _ in range(n_layers)])
    
    @property
    def config(self) -> dict:
        return copy.deepcopy(self._config)
    
    @property
    def device(self) -> torch.device:
        return self.layers[0].centroids.device

    @torch.no_grad()
    def fit_codebooks(self, data: DataLoader):
        is_main_process = os.environ.get('RANK', '0') == '0'
        for i, layer in enumerate(self.layers):
            pbar = tqdm(range(self.max_iters), desc=f"Fitting Layer {i}", disable=not is_main_process, position=0)
            for _ in pbar:
                for batch in data:
                    batch = batch['embedding']
                    for pre_layer in self.layers[:i]:
                        emb, _ = pre_layer(batch, acc=False)
                        batch = batch - emb
                    layer(batch, acc=True)
                shift = layer.update_centroids(data.dataset).max()
                pbar.set_postfix(shift=f'{shift:.6f}')
                if shift < self.tol:
                    break
            pbar.close()
    
    @torch.no_grad()
    def forward(self, x: torch.tensor):
        ids = []
        for layer in self.layers:
            embs, id = layer(x)
            x = x - embs
            ids.append(id)
        reconstruction_loss = torch.norm(x, dim=-1, p=2).mean()
        return RqKmeansOutput(
            sids = torch.stack(ids, dim=1),
            loss_dict = {
                'reconstruction_loss': reconstruction_loss,
            }
        )