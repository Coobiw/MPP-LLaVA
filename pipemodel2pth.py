import torch
from pathlib import Path
import os
from os.path import join
from shutil import copy
import argparse

def convert_model_to_pth(pipeline_model_dir):
    model_state_dict = dict()
    for path in Path(pipeline_model_dir).iterdir():
        print("已经处理文件：{}".format(path))
        if not path.name.startswith('layer'):
            continue
        small_static_dict = torch.load(path, map_location="cpu")
        layer_i = int(path.name.split('-')[0].replace('layer_', ''))
        if layer_i == 0:
            model_state_dict['llm_proj.weight'] = small_static_dict["visionpipe.llm_proj.weight"]
            model_state_dict['llm_proj.bias'] = small_static_dict["visionpipe.llm_proj.bias"]
            model_state_dict["llm_model.transformer.wte.weight"] = small_static_dict["wtepipe.word_embeddings.weight"]
        elif layer_i == 46:
            model_state_dict["llm_model.lm_head.weight"] = small_static_dict["lm_head.weight"]
        elif layer_i == 45:
            model_state_dict["llm_model.transformer.ln_f.weight"] = small_static_dict["final_layernorm.weight"]
        elif layer_i <= 44 and layer_i >=5:
            for k, v in small_static_dict.items():
                model_state_dict["llm_model.transformer." + k.replace("layer",f"h.{layer_i-5}")] = v
        else:
            continue

    torch.save(model_state_dict, join(pipeline_model_dir, "unfreeze_llm_model.pth"))


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='lavis/output/pp_14b/sft/global_step296', type=str, help='')
    return parser.parse_args()


if __name__ == '__main__':
    args = set_args()
    convert_model_to_pth(args.ckpt_dir)