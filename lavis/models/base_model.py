"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os

import numpy as np
import torch
import torch.nn as nn
from lavis.common.dist_utils import download_cached_file, is_dist_avail_and_initialized
from lavis.common.utils import get_abs_path, is_url, get_cache_path
from omegaconf import OmegaConf


class BaseModel(nn.Module):
    """Base class for models."""

    def __init__(self):
        super().__init__()

    @property
    def device(self):
        return list(self.parameters())[0].device

    def load_checkpoint(self, url_or_filename):
        """
        Load from a finetuned checkpoint.

        This should expect no mismatch in the model keys and the checkpoint keys.
        """
        print(f"Checkpoint: {url_or_filename}")
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Build a pretrained model from default configuration file, specified by model_type.

        Args:
            - model_type (str): model type, specifying architecture and checkpoints.

        Returns:
            - model (nn.Module): pretrained or finetuned model, depending on the configuration.
        """
        model_cfg = OmegaConf.load(cls.default_config_path(model_type)).model
        model = cls.from_config(model_cfg)

        return model

    @classmethod
    def default_config_path(cls, model_type):
        assert (
            model_type in cls.PRETRAINED_MODEL_CONFIG_DICT
        ), "Unknown model type {}".format(model_type)
        return get_abs_path(cls.PRETRAINED_MODEL_CONFIG_DICT[model_type])

    def load_checkpoint_from_config(self, cfg, **kwargs):
        """
        Load checkpoint as specified in the config file.

        If load_finetuned is True, load the finetuned model; otherwise, load the pretrained model.
        When loading the pretrained model, each task-specific architecture may define their
        own load_from_pretrained() method.
        """
        load_finetuned = cfg.get("load_finetuned", True)
        load_pretrained = cfg.get("load_pretrained", True)
        if load_finetuned and load_pretrained:
            # load pre-trained weights
            pretrain_path = cfg.get("pretrained", None)
            assert "Found load_finetuned is False, but pretrain_path is None."
            if not os.path.isabs(pretrain_path):
                pretrain_path = get_cache_path(pretrain_path)
            print(f'Start loading pretrained model: {pretrain_path}')
            self.load_from_pretrained(url_or_filename=pretrain_path, **kwargs)

            # load finetuned_weights
            finetune_path = cfg.get("finetuned", None)
            assert (
                finetune_path is not None
            ), "Found load_finetuned is True, but finetune_path is None."
            if not os.path.isabs(finetune_path):
                finetune_path = get_cache_path(finetune_path)
            print(f'Start loading finetuned model: {finetune_path}')
            self.load_checkpoint(url_or_filename=finetune_path)
        elif load_finetuned:
            finetune_path = cfg.get("finetuned", None)
            assert (
                finetune_path is not None
            ), "Found load_finetuned is True, but finetune_path is None."
            if not os.path.isabs(finetune_path):
                finetune_path = get_cache_path(finetune_path)
            print(f'Start loading finetuned model: {finetune_path}')
            self.load_checkpoint(url_or_filename=finetune_path)
        else:
            load_pretrained = cfg.get("load_pretrained", True)
            if load_pretrained:
                # load pre-trained weights
                pretrain_path = cfg.get("pretrained", None)
                assert "Found load_finetuned is False, but pretrain_path is None."
                if not os.path.isabs(pretrain_path):
                    pretrain_path = get_cache_path(pretrain_path)
                print(f'Start loading pretrained model: {pretrain_path}')
                self.load_from_pretrained(url_or_filename=pretrain_path, **kwargs)

    def before_training(self, **kwargs):
        pass

    def get_optimizer_params(self, weight_decay, lr_scale=1):
        p_wd, p_non_wd = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)        
        optim_params = [
            {"params": p_wd, "weight_decay": weight_decay, "lr_scale": lr_scale},
            {"params": p_non_wd, "weight_decay": 0, "lr_scale": lr_scale},
        ]                
        return optim_params
    
    def before_evaluation(self, **kwargs):
        pass

    def show_n_params(self, return_str=True):
        tot = 0
        for p in self.parameters():
            w = 1
            for x in p.shape:
                w *= x
            tot += w
        if return_str:
            if tot >= 1e6:
                return "{:.1f}M".format(tot / 1e6)
            else:
                return "{:.1f}K".format(tot / 1e3)
        else:
            return tot