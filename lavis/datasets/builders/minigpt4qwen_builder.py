"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import warnings
import lavis.common.utils as utils

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry

from lavis.datasets.datasets.minigpt4_instructions import InstructionDataset
from lavis.datasets.datasets.video_instructions import VideoInstructionDataset

@registry.register_builder("minigpt4_instruction")
class Minigpt4InstructionBuilder(BaseDatasetBuilder):
    train_dataset_cls = InstructionDataset
    DATASET_CONFIG_DICT = {
        'default': 'configs/datasets/minigpt4_instruction/defaults_instruction.yaml'
    }

@registry.register_builder("llava_instruction")
class LlavaInstructionBuilder(BaseDatasetBuilder):
    train_dataset_cls = InstructionDataset
    DATASET_CONFIG_DICT = {
        'default': 'configs/datasets/llava_instruction/defaults_instruction.yaml'
    }

@registry.register_builder("llava_pretrain")
class LlavaPretrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = InstructionDataset
    DATASET_CONFIG_DICT = {
        'default': 'configs/datasets/llava_pretrain/defaults.yaml'
    }

@registry.register_builder("llava_instruct_100k")
class LlavaInstuct100KBuilder(BaseDatasetBuilder):
    train_dataset_cls = InstructionDataset
    DATASET_CONFIG_DICT = {
        'default': 'configs/datasets/llava_instruct_100k/defaults.yaml'
    }

@registry.register_builder("llava_instruct_156k")
class LlavaInstuct156KBuilder(BaseDatasetBuilder):
    train_dataset_cls = InstructionDataset
    DATASET_CONFIG_DICT = {
        'default': 'configs/datasets/llava_instruct_156k/defaults.yaml'
    }

@registry.register_builder("videochatgpt_100k")
class VideoChatgpt100KBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoInstructionDataset
    DATASET_CONFIG_DICT = {
        'default': 'configs/datasets/videochatgpt_100k/defaults.yaml'
    }

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)
        max_frames = build_info.get("max_frames")

        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )

            # annotation path
            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            abs_ann_paths = []
            for ann_path in ann_paths:
                if not os.path.isabs(ann_path):
                    ann_path = utils.get_cache_path(ann_path)
                abs_ann_paths.append(ann_path)
            ann_paths = abs_ann_paths

            # visual data storage path
            vis_path = vis_info.storage

            if not os.path.isabs(vis_path):
                # vis_path = os.path.join(utils.get_cache_path(), vis_path)
                vis_path = utils.get_cache_path(vis_path)

            if not os.path.exists(vis_path):
                warnings.warn("storage path {} does not exist.".format(vis_path))

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=ann_paths,
                vis_root=vis_path,
                max_frames=max_frames,
            )

        return datasets