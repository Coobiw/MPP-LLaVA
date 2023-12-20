from lavis.common.registry import registry
from lavis.tasks.deepspeed_base_task import DeepSpeedBaseTask


@registry.register_task("deepspeed_image_text_pretrain")
class DeepSpeedImageTextPretrainTask(DeepSpeedBaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass
