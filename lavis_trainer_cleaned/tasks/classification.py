from tasks.base_task import BaseTask
from common.registry import registry

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from common.logger import MetricLogger, SmoothedValue
from common.registry import registry
from datasets.data_utils import prepare_sample

import logging

@registry.register_task("classification")
class ClassificationTask(BaseTask):
    def __init__(self,**kwargs):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        results = []

        for samples in data_loader:
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def after_evaluation(self, val_result, **kwargs):
        epoch = kwargs.get('epoch',None)
        pred = np.array([], dtype=np.float64)
        labels = np.array([], dtype=np.float64)
        losses = np.array([], dtype=np.float64)
        # import pdb;pdb.set_trace()
        for info in val_result:
            losses = np.append(losses, info[0])
            pred = np.append(pred, info[1])
            labels = np.append(labels, info[2])

        pred = pred.reshape(labels.shape[0],-1)
        acc = (pred.argmax(axis=-1) == labels).sum()
        acc /= labels.shape[0]

        Loss = np.mean(losses)

        if epoch is not None:
            logging.info("EPOCH[{}] -> ACC: {:.6f}, LOSS: {:.6f}".format(epoch, acc * 100, Loss))
        else:
            logging.info("ACC: {:.6f}, LOSS: {:.6f}".format(acc * 100, Loss))

        score = acc * 100

        metrics = {}
        metrics["agg_metrics"] = score
        metrics['ACC'] = acc * 100

        return metrics

    def valid_step(self, model, samples):
        return model.val_step(samples)