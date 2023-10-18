"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from lavis.datasets.datasets.base_dataset import BaseDataset

class Minigpt4QwenDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def collater(self, samples):
        image_list, conversation_list = [], []

        num_answers = []

        for sample in samples:
            image_list.append(sample["image"])
            conversation_list.append(sample["conversations"])

        return {
            "image": torch.stack(image_list, dim=0),
            "conversations": conversation_list,
        }
