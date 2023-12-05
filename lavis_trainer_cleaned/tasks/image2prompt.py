from tasks.base_task import BaseTask
from common.registry import registry

import torch
import torch.nn as nn

@registry.register_task("image2prompt")
class Image2PromptTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass