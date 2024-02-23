"""
对Pipeline里的Layer0做转换，因为layer0 = ViT + Qformer + LLM_Proj + Word Embedding
"""

import torch
import argparse

import os
from pathlib import Path

from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir',type=str,required=True)
args = parser.parse_args()

assert os.path.isdir(args.ckpt_dir), 'need one directory'

ckpt_dir = Path(args.ckpt_dir)

ckpt = torch.load(str(ckpt_dir / 'layer_00-model_states.pt'),map_location='cpu')

llm_proj_weight = ckpt['visionpipe.llm_proj.weight']
llm_proj_bias = ckpt['visionpipe.llm_proj.bias']

model_state_dict = OrderedDict()
model_state_dict['llm_proj.weight'] = llm_proj_weight
model_state_dict['llm_proj.bias'] = llm_proj_bias

torch.save(model_state_dict,str(ckpt_dir/'model.pth'))
