import os
from pathlib import Path

import warnings

import argparse
from omegaconf import OmegaConf

import random
import numpy as np
import torch
import torch.distributed as dist

from common.dist_utils import (
    init_distributed_mode,
    main_process,
)

from common.registry import registry
from common.logger import setup_logger
from tasks import setup_task

from trainer import Trainer

# imports modules for registration
from common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
    ConstantLRScheduler,
)  # 加入到注册表里，不用直接使用（由于是from的import形式，optim.py里的所有类都会加入注册表，所以实际上import一个也可以）

from processors import load_processor
from models import *
from datasets import load_dataset

warnings.filterwarnings('ignore')

def now():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d%H%M")[:-1]

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def get_config(args):
    cfg_path = Path(args.cfg_path)
    assert cfg_path.suffix == '.yaml', 'config file must be .yaml file'
    config = OmegaConf.load(cfg_path)
    init_distributed_mode(config.run)
    return config

def get_transforms(config) -> dict:
    dataset_cfg = config.dataset

    transforms = {}
    transforms['train'] = load_processor(**dataset_cfg.train_cfg.transform)
    transforms['val'] = load_processor(**dataset_cfg.val_cfg.transform)

    return transforms

def get_datasets(config,transforms) -> dict:
    dataset_cfg = config.dataset

    datasets = {}
    train_cfg = dict(dataset_cfg.pop('train_cfg'))
    val_cfg = dict(dataset_cfg.pop('val_cfg'))
    train_cfg['transform'], val_cfg['transform']= transforms['train'],transforms['val']
    datasets["train"] = load_dataset(train_cfg.pop('name'),train_cfg)
    datasets['val'] = load_dataset(val_cfg.pop('name'),val_cfg)

    return datasets

def get_model(config):
    model_cfg = config.model
    model_cls = registry.get_model_class(model_cfg.arch)
    return model_cls.from_config(model_cfg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path',type=str)
    parser.add_argument('--seed',type=int,default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    config = get_config(args)

    setup_logger()

    transforms = get_transforms(config)
    datasets = get_datasets(config,transforms)
    model = get_model(config)
    task = setup_task(config)
    job_id = now()

    trainer = Trainer(config,model,datasets,task,job_id)
    trainer.train()

if __name__ == "__main__":
    main()