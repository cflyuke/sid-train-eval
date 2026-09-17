import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import torch.distributed as dist


@torch.no_grad()
def kmeans(X: torch.Tensor, num_clusters: int, max_iters: int = 1000, tol: float = 1e-4):
    """KMeans算法
    input:
        X: [N, D]
    
    output:
        centroids: [num_clusters, D]
        labels: [N]
    """
    N, D = X.shape
    device = X.device

    # 随机初始化: 当簇数大于样本数时，使用有放回采样保证中心数量正确
    if num_clusters <= N:
        indices = torch.randperm(N, device=device)[:num_clusters]
    else:
        indices = torch.randint(0, N, (num_clusters,), device=device)
    centroids = X[indices].clone()
    i = 0
    while i < max_iters:
        # 计算距离和标签
        distances = torch.cdist(X, centroids, p=2.0)
        labels = torch.argmin(distances, dim=1)

        # 更新聚类中心
        new_centorids = torch.index_add(torch.zeros_like(centroids, dtype=X.dtype, device=X.device), 0, labels, X)
        counts = torch.bincount(labels, minlength=num_clusters).unsqueeze(1).float()
        empty_clusters = (counts.squeeze() == 0)
        if empty_clusters.any():
            num_empty = empty_clusters.sum().item()
            random_points = X[torch.randint(0, N, (num_empty,))]
            new_centorids[empty_clusters] = random_points
            counts[empty_clusters] = 1.0
        new_centorids = new_centorids / counts
        if torch.norm(new_centorids - centroids) < tol:
            break
        centroids = new_centorids
        i += 1

    return centroids, labels

def l2norm(x, dim=-1, eps=1e-12):
    return F.normalize(x, p=2, dim=dim, eps=eps)

class L2NormLayer(nn.Module):
    def __init__(self, dim=-1, eps=1e-12) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x) -> Tensor:
        return l2norm(x, dim=self.dim, eps=self.eps)
    
def batch_to(dataset , device: torch.device):
    return torch.stack([
        torch.as_tensor(item['embedding']) for item in dataset
    ]).to(device)


class GatherWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> tuple[Tensor, ...]:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        out = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(out, x)
        ctx.rank = rank
        return tuple(out)
    
    @staticmethod
    def backward(ctx, *grads: Tensor) -> Tensor:
        stacked = torch.stack(grads)
        dist.all_reduce(stacked)
        return stacked[ctx.rank]
