import os
import json
import glob
from argparse import ArgumentParser

from modules.rqkmeans import RqKmeans, RqKmeansOutput
from modules.evaluate import evaluate
from modules.utils import batch_to

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import set_seed

def get_dataset(data_dir: str, seed: int, normalize: bool):
    dataset = load_dataset('parquet', data_files=glob.glob(os.path.join(data_dir, '*.parquet')), split='train')
    dataset = dataset.shuffle(seed=seed)
    if normalize:
        dataset = dataset.map(lambda x: {'embedding': F.normalize(torch.tensor(x['embedding']), p=2, dim=-1)}, batched=True)
    dataset.set_format(columns=['embedding'], type='torch', output_all_columns=True)
    return dataset

def train(config: dict):
    accelerator = Accelerator(
        project_dir=config['project_dir'],
    )
    set_seed(config['seed'])
    model = RqKmeans(**config['rqkmeans'])
    
    train_dataset = get_dataset(config['train_data'], config['seed'], config['normalize'])
    eval_dataset = get_dataset(config['eval_data'], config['seed'], config['normalize'])

    local_batch_size = config['global_batch_size'] // accelerator.num_processes

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=local_batch_size,
        num_workers=config['num_workers'],
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=local_batch_size,
        num_workers=config['num_workers'],
    )

    model, train_dataloader, eval_dataloader = accelerator.prepare(model, train_dataloader, eval_dataloader)

    accelerator.print('Configurations:\n' + json.dumps(config, indent=2, ensure_ascii=False))
    if config['kmeans_init']:
        accelerator.print('===== Codebook Kmeans Initialization =====')
        assert config['init_datasize'] > 0
        init_dataset = train_dataset.take(config['init_datasize'])
        device = accelerator.device
        init_batch = batch_to(init_dataset, device)
        with torch.no_grad():
            model(init_batch)
            accelerator.wait_for_everyone()

    accelerator.print('===== Fitting RQKmeans Codebooks =====')
    model.fit_codebooks(train_dataloader)

    accelerator.print('===== Evaluating RQKmeans Codebooks =====')
    metrics, collision_ids = evaluate(model, eval_dataloader, config['rqkmeans'], accelerator)
    for key, value in metrics.items():
        accelerator.print(f'[eval]:  {key}: {value:.6f}')
    if accelerator.is_main_process:
        with open(os.path.join(accelerator.project_dir, 'collision_ids.json'), 'w') as f:
            json.dump(collision_ids, f, indent=2)
        with open(os.path.join(accelerator.project_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
    
    accelerator.print('===== Saving Model =====')
    if accelerator.is_main_process:
       state = {
           'model': accelerator.unwrap_model(model).state_dict(),
           'config': accelerator.unwrap_model(model).config,
       }
       torch.save(state, os.path.join(accelerator.project_dir, 'checkpoint.pt'))
    
    accelerator.end_training()
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config/rqkmeans.yaml')
    parser.add_argument('--project_dir', type=str)
    args = parser.parse_args()
    project_dir = args.project_dir
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['project_dir'] = project_dir
    train(config)
