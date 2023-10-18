"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry

from lavis.datasets.datasets.minigpt4_instructions import Minigpt4InstructionDataset

@registry.register_builder("minigpt4_instruction")
class Minigpt4InstructionBuilder(BaseDatasetBuilder):
    train_dataset_cls = Minigpt4InstructionDataset
    DATASET_CONFIG_DICT = {
        'default': 'configs/datasets/minigpt4_instruction/defaults_instruction.yaml'
    }
