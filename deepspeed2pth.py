"""
暂时不支持zero3的转换
因为这里是对`mp_rank_00_model_states.pt`文件进行转换
而zero3由于params shard 不存在这个文件
"""

import torch
import argparse

import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir',type=str,required=True)
args = parser.parse_args()

assert os.path.isdir(args.ckpt_dir), 'need one directory'

ckpt_dir = Path(args.ckpt_dir)

ckpt = torch.load(str(ckpt_dir / 'mp_rank_00_model_states.pt'),map_location='cpu')

model_state_dict = ckpt['module']

torch.save(model_state_dict,str(ckpt_dir/'model.pth'))
