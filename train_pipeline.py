"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import math
import time
from omegaconf import OmegaConf
from functools import partial

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode, init_deepspeed_distributed_mode, is_main_process
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *

from lavis.datasets.data_utils import concat_datasets, reorg_datasets_by_split

from lavis.models.minigpt4qwen_models.minigpt4qwen_pipe import get_model
from deepspeed.pipe import PipelineModule
import deepspeed

import contextlib
from functools import partial

import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--num-stages",type=int,default=0)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls

def collate_fn_minigpt4qwen(batch,preprocess_func,freeze_llm=True,dtype=torch.float32):
    image_list, conversation_list = [], []

    for sample in batch:
        image_list.append(sample["image"])
        conversation_list.append(sample["conversations"])

    new_batch = \
        {
            "image": torch.stack(image_list, dim=0),
            "conversations": conversation_list,
        }
    data_dict = preprocess_func(new_batch['conversations'])

    if not freeze_llm:
        new_batch['image'] = new_batch['image'].to(dtype)

    return ((new_batch['image'], data_dict['input_ids'],data_dict['labels'],data_dict['attention_mask']),
                data_dict['labels']
        )

def get_scheduler(cfg,optimizer,max_steps,steps_per_epoch):
    lr_sched_cls = registry.get_lr_scheduler_class(cfg.run_cfg.lr_sched)

    max_epoch = cfg.run_cfg.max_epoch

    min_lr = cfg.run_cfg.min_lr
    init_lr = cfg.run_cfg.init_lr

    decay_rate = cfg.run_cfg.get("lr_decay_rate", None)
    warmup_start_lr = cfg.run_cfg.get("warmup_lr", -1)
    warmup_steps = int(cfg.run_cfg["warmup_ratio"] * steps_per_epoch) if cfg.run_cfg.get("warmup_ratio",None) else cfg.run_cfg.get("warmup_steps", 0)

    lr_sched = lr_sched_cls(
        optimizer=optimizer,
        max_epoch=max_epoch,
        min_lr=min_lr,
        init_lr=init_lr,
        decay_rate=decay_rate,
        warmup_start_lr=warmup_start_lr,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
    )
    return lr_sched

def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    args = parse_args()
    cfg = Config(args)

    output_dir = cfg.run_cfg.output_dir
    os.makedirs(output_dir,exist_ok=True)

    init_deepspeed_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    ds_cfg = cfg.run_cfg.deepspeed_config

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    # import pdb;pdb.set_trace()
    datasets = reorg_datasets_by_split(datasets)
    datasets = concat_datasets(datasets)

    model = task.build_model(cfg)
    freeze_llm = model.freeze_llm

    # preprocoss of multimodal tokenizer
    preprocess_func = \
        partial(model.preprocess,tokenizer=model.llm_tokenizer,max_len=model.max_txt_len,image_len=model.num_query_token)
    collate_fn_minigpt4qwen_func = partial(collate_fn_minigpt4qwen, preprocess_func=preprocess_func)
    
    assert args.num_stages > 1, f'pipeline parallel need stages more than 1, current num_stages is {args.num_stages}'

    model = PipelineModule(layers=get_model(model,freeze_llm=freeze_llm), num_stages=args.num_stages, partition_method='uniform')# if freeze_llm else 'parameters')

    print_string = f'GPU{cfg.run_cfg.gpu}\t' + f'Trainable Params: {sum([param.numel() for _, param in model.named_parameters() if param.requires_grad])}'
    os.system(f'echo {print_string}')

    model.cuda().bfloat16()

    engine, optimizer, _, _ = deepspeed.initialize(
                        model=model,
                        config=OmegaConf.to_container(ds_cfg),
                        model_parameters=[p for p in model.parameters() if p.requires_grad],
                    )
    model_dtype = next(model.parameters()).dtype

    g = torch.Generator()
    sampler = torch.utils.data.distributed.DistributedSampler(
                datasets['train'],
                num_replicas=engine.dp_world_size,
                rank=engine.mpu.get_data_parallel_rank(),
                shuffle=False
            )
    # print_string = f'GPU{cfg.run_cfg.gpu}\t' + f'rank{engine.mpu.get_data_parallel_rank()}'
    # os.system(f'echo {print_string}')

    train_dataloader = DataLoader(datasets['train'],
                            shuffle=False,
                            drop_last=True,
                            batch_size=ds_cfg.train_micro_batch_size_per_gpu,
                            generator=g,
                            sampler=sampler,
                            collate_fn=partial(collate_fn_minigpt4qwen_func,freeze_llm=freeze_llm,dtype=torch.float32 if freeze_llm else model_dtype),
                            num_workers=cfg.run_cfg.num_workers,
                        )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / ds_cfg.gradient_accumulation_steps)
    print(num_update_steps_per_epoch)

    train_dataloader = deepspeed.utils.RepeatingLoader(train_dataloader)

    lr_scheduler = get_scheduler(cfg,optimizer,
                            max_steps=cfg.run_cfg.max_epoch * num_update_steps_per_epoch,
                            steps_per_epoch=num_update_steps_per_epoch
                        )

    start = time.time()
    all_loss = 0.0

    if is_main_process():
        wandb.init(project="MBPP-Qwen14B")
    
    for epoch in range(cfg.run_cfg.max_epoch):
        sampler.set_epoch(epoch)

        train_iter = iter(train_dataloader)
        for cur_step in range(num_update_steps_per_epoch):
            step = cur_step + epoch * num_update_steps_per_epoch
            with (torch.cuda.amp.autocast(dtype=model_dtype,cache_enabled=False) if freeze_llm and (model_dtype != torch.float32) else contextlib.nullcontext()):
                loss = engine.train_batch(data_iter=train_iter)
            
            lr_scheduler.step(cur_epoch=epoch, cur_step=step)

            print(f"step = {step}, loss = {loss.item()}, lr={optimizer.param_groups[0]['lr']}")

            if is_main_process():
                wandb.log({"loss": loss.item()}, step=step)
                wandb.log({"learning_rate": optimizer.param_groups[0]['lr']}, step=step)
            
            all_loss += loss.item()
            if (step + 1) % cfg.run_cfg.log_freq == 0:
                now_time = time.time()
                avg_time = (now_time - start) / cfg.run_cfg.log_freq
                avg_loss = all_loss / cfg.run_cfg.log_freq
                print(f"Step={step:>6}, lr={optimizer.param_groups[0]['lr']}, loss={avg_loss:.4f}, {avg_time:.2f} it/s")
                start = now_time
                all_loss = 0.0

            if (step + 1) % num_update_steps_per_epoch == 0:
                print(f"Saving at step {step}")
                engine.save_checkpoint(output_dir)
    
    if is_main_process():
        wandb.finish()



if __name__ == "__main__":
    main()
