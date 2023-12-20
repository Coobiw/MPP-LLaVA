"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.tasks.image_text_pretrain import ImageTextPretrainTask
from lavis.tasks.deepspeed_base_task import DeepSpeedBaseTask
from lavis.tasks.deepspeed_image_text_pretrain import DeepSpeedImageTextPretrainTask


def setup_task(cfg):
    assert "task" in cfg.run_cfg, "Task name must be provided."

    task_name = cfg.run_cfg.task
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task


__all__ = [
    "BaseTask",
    "ImageTextPretrainTask",
    "DeepSpeedBaseTask",
    "DeepSpeedImageTextPretrainTask",
]
