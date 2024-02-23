"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import math
import time
from omegaconf import OmegaConf

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode, init_deepspeed_distributed_mode
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

def collate_fn_minigpt4qwen(batch,preprocess_func):
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

    return ((new_batch['image'], data_dict['input_ids'],data_dict['labels'],data_dict['attention_mask']),
                data_dict['labels']
        )

def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    args = parse_args()
    cfg = Config(args)

    output_dir = cfg.run_cfg.output_dir

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
    # preprocoss of multimodal tokenizer
    preprocess_func = \
        partial(model.preprocess,tokenizer=model.llm_tokenizer,max_len=model.max_txt_len,image_len=model.num_query_token)
    collate_fn_minigpt4qwen_func = partial(collate_fn_minigpt4qwen, preprocess_func=preprocess_func)
    
    assert args.num_stages > 1, f'pipeline parallel need stages more than 1, current num_stages is {args.num_stages}'

    model = PipelineModule(layers=get_model(model), num_stages=args.num_stages, partition_method='uniform')
    print_string = f'GPU{cfg.run_cfg.gpu}\t' + f'Trainable Params: {sum([param.numel() for _, param in model.named_parameters() if param.requires_grad])}'
    os.system(f'echo {print_string}')
    model.cuda().bfloat16()


    g = torch.Generator()
    train_dataloader = DataLoader(datasets['train'],
                            shuffle=True,
                            drop_last=True,
                            batch_size=ds_cfg.train_micro_batch_size_per_gpu,
                            generator=g,
                            collate_fn=collate_fn_minigpt4qwen_func,
                        )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / ds_cfg.gradient_accumulation_steps)
    print(num_update_steps_per_epoch)

    train_dataloader = deepspeed.utils.RepeatingLoader(train_dataloader)

    engine, _, _, _ = deepspeed.initialize(model=model, config=OmegaConf.to_container(ds_cfg), model_parameters=[p for p in model.parameters() if p.requires_grad])
    model_dtype = next(model.parameters()).dtype

    start = time.time()
    all_loss = 0.0
    train_iter = iter(train_dataloader)

    for step in range(cfg.run_cfg.max_epoch * num_update_steps_per_epoch):
        with (torch.cuda.amp.autocast(dtype=model_dtype,cache_enabled=False) if model_dtype != torch.float32 else contextlib.nullcontext()):
            loss = engine.train_batch(data_iter=train_iter)

        print("step = {}, loss = {}".format(step, loss.item()))
        all_loss += loss.item()
        if (step + 1) % cfg.run_cfg.log_freq == 0:
            now_time = time.time()
            avg_time = (now_time - start) / cfg.run_cfg.log_freq
            avg_loss = all_loss / cfg.run_cfg.log_freq
            print(f"Step={step:>6}, loss={avg_loss:.4f}, {avg_time:.2f} it/s")
            start = now_time
            all_loss = 0.0

        if (step + 1) % num_update_steps_per_epoch == 0:
            print(f"Saving at step {step}")
            engine.save_checkpoint(output_dir)



if __name__ == "__main__":
    main()
