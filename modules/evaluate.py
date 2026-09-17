import random
from typing import Optional, List, Dict

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets import Dataset
import numpy as np


def get_metrics(layer_ids: Tensor, codebook_size: int) -> dict:
    """评估生成的sid的质量
    input:
        layer_ids: [B, n_layers]
        codebook_size: int

    output:
        metrics: dict 包含各层及整体的利用率和基尼系数等
    """
    metrics = {}
    total_samplers, n_layers = layer_ids.shape
    for layer_idx in range(n_layers):
        current_layer_ids = layer_ids[:, layer_idx]

        unique_codes = torch.unique(current_layer_ids).numel()

        # 码本利用率 cur：激活的独立编码数 / 码本大小 
        cur = unique_codes / codebook_size

        # 码本分布熵 entropy：- \sum p_i log(p_i)，p_i是第i个编码被激活的概率
        code_counts = torch.bincount(current_layer_ids.flatten(), minlength=codebook_size)
        code_probs = (code_counts.float() / code_counts.sum()).clamp(min=1e-8)
        entropy = -(code_probs * torch.log(code_probs)).sum()

        # 基尼系数 gini：2 / n \sum_i (i / n - \sum_{j = 1}^i p_j)
        code_probs_sorted = torch.sort(code_probs)[0]
        gini_coeff = 2 / codebook_size  * ((codebook_size + 1) / 2 - code_probs_sorted.cumsum(dim=0).sum())

        metrics[f'layer_{layer_idx + 1}_cur'] = cur
        metrics[f'layer_{layer_idx + 1}_gini'] = gini_coeff
        metrics[f'layer_{layer_idx + 1}_entropy'] = entropy

    # 2. 计算整体 (sid_1, sid_2, ..., sid_n) 组合利用率
    full_seqs = layer_ids.cpu().numpy()
    unique_seqs, seq_counts = np.unique(full_seqs, axis=0, return_counts=True)
    total_unique_sids = len(unique_seqs)

    # 整体cur
    total_cur = total_unique_sids / (codebook_size ** n_layers)

    # ID collision
    id_collision = (seq_counts * (seq_counts - 1)).sum() / (total_samplers * (total_samplers - 1))

    # 计算整体基尼系数，由于码本数量过大，只考虑样本间均匀分布
    seq_counts = np.concatenate([
        seq_counts,
        np.zeros(total_samplers - total_unique_sids, dtype=seq_counts.dtype),
    ])
    seq_probs = np.sort(seq_counts) / total_samplers
    global_gini_coeff = 2 / total_samplers * ((total_samplers + 1) / 2 - np.cumsum(seq_probs).sum())

    metrics['global_unique_sids'] = total_unique_sids
    metrics['global_id_collision'] = id_collision
    metrics['global_cur'] = total_cur
    metrics['global_gini'] = global_gini_coeff

    return metrics



def _flatten_objects(objects):
    if isinstance(objects, (list, tuple)):
        flattened = []
        for item in objects:
            nested = _flatten_objects(item)
            if isinstance(nested, list):
                flattened.extend(nested)
            else:
                flattened.append(nested)
        return flattened
    elif isinstance(objects, dict):
        return {key: _flatten_objects(value) for key, value in objects.items()}
    else:
        return objects
    
def convert_to_serializable(obj):
    """递归转换 tensor 和其他不可序列化对象为 Python native types"""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def get_collision_ids(ids: List[str], sids: Tensor) -> Dict[str, List[str]]:
    unique_rows, inverse, counts = torch.unique(
        sids, dim=0, return_inverse=True, return_counts=True
    )
    collision_group_indices = torch.where(counts > 1)[0]
    collision_ids = dict()
    for idx in collision_group_indices:
        group_num_pos = torch.where(inverse == idx)[0].tolist()
        group_ids = [ids[i] for i in group_num_pos]
        collision_ids[str(unique_rows[idx].tolist())] = group_ids
    return collision_ids

def get_collision_similarity(collision_ids: Dict[str, List[str]], dataset: Dataset) -> Dict[str, float]:
    id_pairs = []
    for sid, ids in collision_ids.items():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_pairs.append((ids[i], ids[j]))
    if not id_pairs:
        return {'similarity_mean': 0.0, 'similarity_std': 0.0}
    id_pairs = random.sample(id_pairs, min(len(id_pairs), 1000))
    id1 = [pair[0] for pair in id_pairs]
    id2 = [pair[1] for pair in id_pairs]
    id_set = set(id1) | set(id2)
    dataset_filtered = dataset.filter(lambda x: x['id'] in id_set)
    id_to_embedding = {item['id']: item['embedding'] for item in dataset_filtered}
    valid_pairs = [(a, b) for a, b in zip(id1, id2) if a in id_to_embedding and b in id_to_embedding]
    if not valid_pairs:
        return {'similarity_mean': 0.0, 'similarity_std': 0.0}
    embeddings1 = torch.stack([torch.as_tensor(id_to_embedding[a]) for a, _ in valid_pairs])
    embeddings2 = torch.stack([torch.as_tensor(id_to_embedding[b]) for _, b in valid_pairs])
    similarities = F.cosine_similarity(embeddings1, embeddings2)
    return {'similarity_mean': similarities.mean().item(), 'similarity_std': similarities.std().item()}

def evaluate(model: nn.Module, eval_dataloader: DataLoader, config: dict, accelerator: Optional[Accelerator] = None) -> dict:
    model_train = model.training
    model.eval()

    with torch.no_grad():
        all_ids = []
        all_sids = []
        all_recon_loss = []
        for batch in eval_dataloader:
            embeddings = batch['embedding']
            embeddings = embeddings.to(model.device)

            output = model(embeddings)
            recon_loss = ((output.reconstruction - embeddings)**2).sum(dim=-1).mean()
            if accelerator is not None:
                all_ids.append(accelerator.gather_for_metrics(batch['id'], use_gather_object=True))
                all_sids.append(accelerator.gather_for_metrics(output.sids))
                all_recon_loss.append(accelerator.gather_for_metrics(recon_loss))
            else:
                all_ids.append(batch['id'])
                all_sids.append(output.sids)
                all_recon_loss.append(recon_loss)
        metrics = {}
        metrics['val/reconstruction_loss'] = torch.stack(all_recon_loss).mean().item()

        all_sids = torch.cat(all_sids, dim=0)
        sid_metrics = get_metrics(all_sids, config['codebook_size'])

        all_ids = _flatten_objects(all_ids)
        n_layers = all_sids.shape[-1]
        collision_ids = {}
        for i in range(n_layers):
            layer_ids = all_sids[:, :i+1]
            layer_collision_ids = get_collision_ids(all_ids, layer_ids)
            collision_ids[f'pre_{i + 1}_layer'] = layer_collision_ids
            collision_similarity = get_collision_similarity(layer_collision_ids, eval_dataloader.dataset)
            for key, value in collision_similarity.items():
                metrics[f'val/pre_{i + 1}_layer_{key}'] = value

        for key, value in sid_metrics.items():
            metrics[f'val/{key}'] = value
    if model_train:
        model.train()
    return convert_to_serializable(metrics), convert_to_serializable(collision_ids)
