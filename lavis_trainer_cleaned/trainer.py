import os
import shutil
from pathlib import Path

import logging
import json

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets.dataloader_utils import get_dataloaders
from common.dist_utils import main_process, is_main_process, is_dist_avail_and_initialized
from common.registry import registry
from omegaconf import OmegaConf

import time
import datetime

from copy import deepcopy

class Trainer:
    def __init__(self,config,model,datasets,task,job_id):
        self.config = config
        self.job_id = job_id
        self._model = model
        self.datasets = datasets
        self.task = task

        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._scaler = None
        self._dataloaders = None
        self._lr_sched = None

        self.start_epoch = 0

        self.setup_output_dir()

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self.config.run.device)

        return self._device

    @property
    def use_distributed(self):
        return self.config.run.distributed

    @property
    def model(self):
        """
        A property to get the DDP-wrapped model on the device.
        """
        # move model to device
        if self._model.device != self.device:
            self._model = self._model.to(self.device)

            # ddp training wrapper
            if self.use_distributed:
                if self._wrapped_model is None:
                    self._wrapped_model = DDP(
                        self._model, device_ids=[self.config.run.gpu]
                    )
            else:
                self._wrapped_model = self._model

        return self._wrapped_model

    @property
    def dataloaders(self) -> dict:
        run_cfg = self.config.run
        if self._dataloaders is None:
            self._dataloaders = get_dataloaders(
                datasets = self.datasets,
                batch_size = run_cfg.batch_size,
                batch_size_val = run_cfg.batch_size_val,
                num_worker = run_cfg.num_worker,
                ddp = run_cfg.distributed,
            )
        return self._dataloaders

    @property
    def optimizer(self):
        if self._optimizer is None:
            # 可以用这个实现逐层lr decay
            # 需要重写model的get_optimizer_params,可以参考lavis的vit
            lr_scale = self.config.run.get("lr_layer_decay", 1)
            weight_decay = self.config.run.get("weight_decay", 0.05)
            optim_params = self._model.get_optimizer_params(weight_decay,lr_scale)

            num_parameters = 0
            for p_group in optim_params:
                for p in p_group["params"]:
                    num_parameters += p.data.nelement()
            logging.info("number of trainable parameters: {}".format(num_parameters))

            beta2 = self.config.run.get("beta2", 0.999)

            self._optimizer = torch.optim.AdamW(
                optim_params,
                lr=float(self.config.run.init_lr),
                betas=(0.9, beta2),
            )
        return self._optimizer

    @property
    def scaler(self):
        amp = self.config.run.get("amp", False)

        if amp:
            if self._scaler is None:
                self._scaler = torch.cuda.amp.GradScaler()

        return self._scaler

    @property
    def lr_scheduler(self):
        """
        A property to get and create learning rate scheduler by split just in need.
        """
        if self._lr_sched is None:
            lr_sched_cls = registry.get_lr_scheduler_class(self.config.run.lr_sched)

            # max_epoch = self.config.run.max_epoch
            max_epoch = self.max_epoch
            # min_lr = self.config.run.min_lr
            min_lr = self.min_lr
            # init_lr = self.config.run.init_lr
            init_lr = self.init_lr

            # optional parameters
            decay_rate = self.config.run.get("lr_decay_rate", None)
            warmup_start_lr = self.config.run.get("warmup_lr", -1)
            warmup_steps = self.config.run.get("warmup_steps", 0)

            self._lr_sched = lr_sched_cls(
                optimizer=self.optimizer,
                max_epoch=max_epoch,
                min_lr=min_lr,
                init_lr=init_lr,
                decay_rate=decay_rate,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
            )

        return self._lr_sched

    @property
    def cuda_enabled(self):
        return self.device.type == "cuda"

    @property
    def max_epoch(self):
        return int(self.config.run.max_epoch)

    @property
    def log_freq(self):
        log_freq = self.config.run.get("log_freq", 50)
        return int(log_freq)

    @property
    def init_lr(self):
        return float(self.config.run.init_lr)

    @property
    def min_lr(self):
        return float(self.config.run.min_lr)

    @property
    def accum_grad_iters(self):
        return int(self.config.run.get("accum_grad_iters", 1))

    @property
    def grad_norm_clip(self):
        return self.config.run.get("grad_norm_clip",None)

    @property
    def evaluate_only(self):
        """
        Set to True to skip training.
        """
        return self.config.run.evaluate

    @property
    def eval_freq(self):
        """
        Set the frequency of evaluating
        """
        return self.config.run.get("eval_freq",1)
    
    @property
    def save_freq(self):
        """
        Set the frequency of evaluating
        """
        return self.config.run.get("save_freq",1)

    @property
    def resume_ckpt_path(self):
        return self.config.run.get("resume_ckpt_path", None)

    @property
    def train_loader(self):
        train_dataloader = self.dataloaders["train"]

        return train_dataloader

    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model

    def setup_output_dir(self):
        output_dir = Path(self.config.run.output_dir) / self.job_id
        result_dir = output_dir / "result"

        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        self.result_dir = result_dir
        self.output_dir = output_dir

    @main_process
    def _save_checkpoint(self, cur_epoch, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]

        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": OmegaConf.to_container(self.config),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)

    def _reload_best_model(self, model):
        """
        Load the best checkpoint for evaluation.
        """
        checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")

        logging.info("Loading checkpoint from {}.".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as e:
            logging.warning(
                """
                Key mismatch when loading checkpoint. This is expected if only part of the model is saved.
                Trying to load the model with strict=False.
                """
            )
            model.load_state_dict(checkpoint["model"], strict=False)
        return model

    def _load_checkpoint(self, filename):
        """
        Resume from a checkpoint.
        """
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location=self.device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        self.unwrap_dist_model(self.model).load_state_dict(state_dict)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = checkpoint["epoch"] + 1
        logging.info("Resume checkpoint from {}".format(filename))

    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(OmegaConf.to_container(self.config), indent=4) + "\n")

    @torch.no_grad()
    def eval_epoch(self, cur_epoch, skip_reload=False):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, we will use provided weights and skip reloading the best checkpoint .
        """
        data_loader = self.dataloaders.get('val', None)
        assert data_loader, "data_loader for split {} is None.".format("val")

        # TODO In validation, you need to compute loss as well as metrics
        # TODO consider moving to model.before_evaluation()
        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        self.task.before_evaluation(
            model=model,
            dataset=self.datasets["val"],
        )
        results = self.task.evaluation(model, data_loader)

        if results is not None:
            return self.task.after_evaluation(
                val_result=results,
                epoch=cur_epoch,
            )

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_epoch = 0
        best_metrics = {}

        self.log_config()

        # resume from checkpoint if specified
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            # training phase
            if not self.evaluate_only:
                logging.info("Start training")
                # See https://github.com/salesforce/LAVIS/issues/449
                # if cur_epoch == self.start_epoch:
                #     self.task.before_training(
                #         model=self.unwrap_dist_model(self.model),
                #         dataset=self.datasets["train"],
                #     )
                train_stats = self.train_epoch(cur_epoch)
                self.log_stats(split_name="train", stats=train_stats)

            # evaluation phase
            if cur_epoch % self.eval_freq == 0 or cur_epoch == self.max_epoch -1:
                logging.info("Evaluating on {}.".format("val"))

                val_log = self.eval_epoch(
                    cur_epoch=cur_epoch,
                )
                if val_log is not None:
                    if is_main_process():
                        assert (
                            "agg_metrics" in val_log
                        ), "No agg_metrics found in validation log."

                        agg_metrics = val_log["agg_metrics"]
                        if agg_metrics > best_agg_metric:
                            best_epoch, best_agg_metric = cur_epoch, agg_metrics
                            best_metrics = deepcopy(val_log)

                            self._save_checkpoint(cur_epoch, is_best=True)

                        if cur_epoch % self.save_freq == 0 or cur_epoch == self.max_epoch -1:
                            self._save_checkpoint(cur_epoch, is_best=False)
                        val_log.update({"best_epoch": best_epoch})
                        self.log_stats(val_log, "val")
                else:  # 没有定义task的evaluation
                    if cur_epoch % self.save_freq == 0 or cur_epoch == self.max_epoch -1:
                        self._save_checkpoint(cur_epoch, is_best=False)

            else:
                if not self.evaluate_only:
                    if cur_epoch % self.save_freq == 0:
                        self._save_checkpoint(cur_epoch, is_best=False)

            if self.evaluate_only:
                break

            if is_dist_avail_and_initialized():
                dist.barrier()

        return best_metrics

    def train_epoch(self, epoch):
        # train
        self.model.train()

        return self.task.train_epoch(
            epoch=epoch,
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq,
            accum_grad_iters=self.accum_grad_iters,
            grad_norm_clip=self.grad_norm_clip,
        )