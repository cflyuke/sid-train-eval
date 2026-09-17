import os
import glob
import json
from argparse import ArgumentParser

from modules.rqvae import RqVae, RqVaeOutput, RqVaeConfig
from modules.loss import RqVaeCriterion
from modules.utils import batch_to
from modules.evaluate import evaluate

import yaml
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def get_dataset(data_dir: str, seed: int, normalize: bool):
    dataset = load_dataset('parquet', data_files=glob.glob(os.path.join(data_dir, '*.parquet')), split='train')
    dataset = dataset.shuffle(seed=seed)
    if normalize:
        dataset = dataset.map(lambda x: {'embedding': F.normalize(torch.tensor(x['embedding']), p=2, dim=-1)}, batched=True)
    dataset.set_format(columns=['embedding'], type='torch', output_all_columns=True)
    return dataset


def get_pair_dataset(data_dir: str, seed: int, normalize: bool):
    dataset = load_dataset('parquet', data_files=glob.glob(os.path.join(data_dir, '*.parquet')), split='train')
    dataset = dataset.shuffle(seed=seed)
    if normalize:
        dataset = dataset.map(lambda x: {
            'embedding_a': F.normalize(torch.tensor(x['embedding_a']), p=2, dim=-1),
            'embedding_b': F.normalize(torch.tensor(x['embedding_b']), p=2, dim=-1),
        }, batched=True)
    dataset.set_format(columns=['embedding_a', 'embedding_b'], type='torch', output_all_columns=True)
    return dataset


def train(config: dict):
    # 使用 Accelerate 进行数据并行
    accelerator = Accelerator(
        mixed_precision=config['mixed_precision'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        log_with=config['log_with'],
        project_dir=config['project_dir'],
    )
    accelerator.init_trackers(f'{config["log_with"]}_logs')
    set_seed(config['seed'])

    # 配置模型和优化器

    # 是否需要进行续训，加载对应step时的optimizer和model state
    resume_epoch = epoch_step = global_step = 0
    if config['checkpoint_path'] is not None:
        checkpoint = torch.load(config['checkpoint_path'], map_location=accelerator.device, weights_only=False)
        config['model'] = checkpoint['config']
        model = RqVae.from_config(checkpoint['config'])
        model.load_state_dict(checkpoint['model'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        resume_epoch = checkpoint['epoch']
        epoch_step = checkpoint['epoch_step']
        global_step = checkpoint['global_step']
        accelerator.print(f'Resumed from checkpoint: {config["checkpoint_path"]}')
        accelerator.print(f'Resume epoch: {resume_epoch}, epoch_step: {epoch_step}, global_step: {global_step}')
    else:
        model = RqVae.from_config(config['model'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    # 实例化criterion（独立于model，不影响checkpoint）
    criterion = RqVaeCriterion.from_config(config['criterion'])
    
    # 配置数据
    use_contrastive = config.get('use_contrastive', False)
    train_dataset = get_dataset(config['train_data'], config['seed'], config['normalize'])
    eval_dataset = get_dataset(config['eval_data'], config['seed'], config['normalize'])

    # CL pair 数据（可选）
    cl_dataloader = None
    if use_contrastive:
        cl_dataset = get_pair_dataset(config['cl_data'], config['seed'], config['normalize'])
        cl_interval = config.get('cl_interval', 5)  # 每隔 cl_interval 个 epoch 跑一轮 CL

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
    if use_contrastive:
        cl_dataloader = DataLoader(
            cl_dataset,
            batch_size=local_batch_size,
            num_workers=config['num_workers'],
        )
        model, optimizer, train_dataloader, cl_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, cl_dataloader, eval_dataloader
        )
    else:
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

    accelerator.print('Configurations:\n' + json.dumps(config, indent=2, ensure_ascii=False))

    # 加载同一批数据，对所有机器进行模型参数初始化
    if config['model']['codebook_kmeans_init'] and config['checkpoint_path'] is None:
        accelerator.print('===== Codebook Kmeans Initialization =====')
        assert config['init_datasize'] > 0
        device = accelerator.device
        init_dataset = train_dataset.take(config['init_datasize'])
        init_batch = torch.stack([
            torch.as_tensor(item['embedding']) for item in init_dataset
        ]).to(device)
        with torch.no_grad():
            model(init_batch)
            accelerator.wait_for_everyone()

    # 初始的时候评估一下
    accelerator.print(f"===== Initialization Metrics=====")
    eval_metrics, collision_ids = evaluate(model, eval_dataloader, config['model'], accelerator)
    for key, value in eval_metrics.items():
        accelerator.print(f"[eval]:  {key}: {value:.6f}")
    accelerator.log(eval_metrics, step=global_step)

    # 开始训练
    accelerator.print(f"===== Start Training =====")
    ep = resume_epoch
    for ep in range(resume_epoch, config['epochs']):
        activeloader = train_dataloader
        model.train()

        # 断点续训，从对应step开始，避免重复训练
        if epoch_step and ep == resume_epoch:
            activeloader = accelerator.skip_first_batches(train_dataloader, epoch_step * config['gradient_accumulation_steps'])

        # ===== 正常训练 epoch =====
        for batch in activeloader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                embeddings = batch['embedding']
                output = model(embeddings)
                loss_dict = criterion(output, embeddings, use_contrastive=False)
                loss = loss_dict['loss']
                accelerator.backward(loss)
                accelerator.wait_for_everyone()
                optimizer.step()
                accelerator.wait_for_everyone()
                if accelerator.sync_gradients:
                    epoch_step += 1
                    global_step += 1
                    if global_step % config['log_step'] == 0:
                        reconstruction_loss = accelerator.reduce(loss_dict['reconstruction_loss'], reduction="mean")
                        accelerator.print(f"ep: {ep}, step: {global_step}, reconstuction loss: {reconstruction_loss.item()}")
                        accelerator.log({"train/reconstruction_loss": reconstruction_loss.item()}, step=global_step)
                    if global_step % config['eval_step'] == 0:
                        eval_metrics, collision_ids = evaluate(model, eval_dataloader, config['model'], accelerator)
                        accelerator.print(f"===== Evaluation at step {global_step} =====")
                        for key, value in eval_metrics.items():
                            accelerator.print(f"[eval]:  {key}: {value:.6f}")
                        accelerator.log(eval_metrics, step=global_step)
                    if global_step % config['save_step'] == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            unwrapped_model = accelerator.unwrap_model(model)
                            state = {
                                'epoch': ep,
                                'epoch_step': epoch_step,
                                'global_step': global_step,
                                'model': unwrapped_model.state_dict(),
                                'config': unwrapped_model.config,
                                'optimizer': optimizer.state_dict(),
                            }
                            torch.save(state, os.path.join(accelerator.project_dir, f'checkpoint_epoch_{ep}_step_{epoch_step}.pt'))

        # ===== CL epoch（每隔 cl_interval 个 epoch 跑一轮对比学习）=====
        if use_contrastive and (ep + 1) % cl_interval == 0:
            accelerator.print(f"===== CL Epoch after ep {ep} =====")
            model.train()
            for cl_batch in cl_dataloader:
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    cl_embeddings = torch.cat([cl_batch['embedding_a'], cl_batch['embedding_b']], dim=0)
                    cl_output = model(cl_embeddings)
                    cl_loss_dict = criterion(cl_output, cl_embeddings, use_contrastive=True)
                    cl_loss = cl_loss_dict['loss']
                    accelerator.backward(cl_loss)
                    accelerator.wait_for_everyone()
                    optimizer.step()
                    accelerator.wait_for_everyone()
                    if accelerator.sync_gradients:
                        global_step += 1
                        if global_step % config['log_step'] == 0:
                            reconstruction_loss = accelerator.reduce(cl_loss_dict['reconstruction_loss'], reduction="mean")
                            accelerator.print(f"[CL] ep: {ep}, step: {global_step}, reconstruction loss: {reconstruction_loss.item()}")
                            accelerator.log({"train/cl_reconstruction_loss": reconstruction_loss.item()}, step=global_step)

            # CL epoch 结束后评估一次
            eval_metrics, collision_ids = evaluate(model, eval_dataloader, config['model'], accelerator)
            accelerator.print(f"===== Evaluation after CL epoch (step {global_step}) =====")
            for key, value in eval_metrics.items():
                accelerator.print(f"[eval]:  {key}: {value:.6f}")
            accelerator.log(eval_metrics, step=global_step)

        epoch_step = 0  # 每个 epoch 结束重置
    
    # save after training
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        state = {
            'epoch': ep,
            'epoch_step': epoch_step,
            'global_step': global_step,
            'model': unwrapped_model.state_dict(),
            'config': unwrapped_model.config,
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, os.path.join(accelerator.project_dir, f'checkpoint_final.pt'))

    if accelerator.is_main_process:
        with open(os.path.join(accelerator.project_dir, 'collision_ids.json'), 'w') as f:
            json.dump(collision_ids, f, indent=2)
        with open(os.path.join(accelerator.project_dir, 'metrics.json'), 'w') as f:
            json.dump(eval_metrics, f, indent=2)
    accelerator.end_training()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('--config', type=str, default='config/rqvae.yaml')
    args.add_argument('--project_dir', type=str)
    args = args.parse_args()
    project_dir = args.project_dir
    config = load_config(args.config)
    config['project_dir'] = project_dir
    train(config)