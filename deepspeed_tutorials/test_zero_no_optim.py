import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

import time
import os
import argparse

import deepspeed
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

from torch.utils.data import Dataset

from eva_vit import create_eva_vit_g

from tqdm import tqdm
from scheduler import CosineAnnealingLR

def init_seed(seed=999):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_ckpt(model_engine, scheduler, ckpt_dir, ckpt_tag):
    _, client_sd = model_engine.load_checkpoint(load_dir=ckpt_dir,tag=ckpt_tag)

    global_step = client_sd['global_step']
    start_epoch = client_sd['epoch']
    scheduler.load_state_dict(client_sd['scheduler'])
    
    return start_epoch, global_step

class ExampleDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = [torch.randn(3,224,224,device="cpu") for _ in range(400)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

if __name__ == "__main__":
    init_seed()

    save_interval = 1
    save_dir = './ckpt_zero3_value-2e8'
    # save_dir = './ckpt_zero2'

    # ckpt_dir = './ckpt_zero2'
    ckpt_dir = None
    ckpt_tag = None


    parser = argparse.ArgumentParser(description='Args')
    parser.add_argument('--deepspeed_config', default=None, type=str)
    parser.add_argument('--local_rank',default=-1,type=int)

    args, _unknown = parser.parse_known_args()

    deepspeed.init_distributed(
        dist_backend='nccl',
        init_method='env://',
        distributed_port=8080,
    )

    if args.deepspeed_config is not None and args.local_rank == -1:
        print('use environment local_rank')
        args.local_rank = int(os.environ.get('LOCAL_RANK', -1))

    torch.cuda.set_device(args.local_rank)

    # deepspeed.utils.dist.barrier()
    dist.barrier()

    if args.local_rank == 0:
        print('unknown args:',_unknown)
    
    # deepspeed.utils.dist.barrier()
    dist.barrier()

    print(f"Local rank: {args.local_rank}")


    model = create_eva_vit_g(img_size=224,drop_path_rate=0.,use_checkpoint=False,precision="fp32").cuda()
    trainset = ExampleDataset()

    # optimizer = optim.AdamW(model.parameters(),lr=1e-4,eps=1e-7,weight_decay=0.)
    # 自定义optimizer使用zero会报错
    """
    deepspeed.runtime.zero.utils.ZeRORuntimeException: 
    You are using ZeRO-Offload with a client provided optimizer (<class 'torch.optim.adamw.AdamW'>) 
    which in most cases will yield poor performance.
    Please either use deepspeed.ops.adam.DeepSpeedCPUAdam or set an optimizer in your ds-config. 
    If you really want to use a custom optimizer w. 
    ZeRO-Offload and understand the performance impacts you can also set <"zero_force_ds_cpu_optimizer": false> in your configuration file.
    """


    model_engine, optimizer , trainloader , _ = deepspeed.initialize(
        args,
        model=model,
        model_parameters=model.parameters(),
        training_data=trainset,
        # optimizer=optimizer,
    )

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=10,eta_min=1e-5)
    # 使用torch自定义的scheduler，会报错
    """
    TypeError: DeepSpeedZeroOptimizer is not an Optimizer
    """

    scheduler = CosineAnnealingLR(optimizer=optimizer,T_max=10,base_lr=1e-4,eta_min=1e-5)

    if ckpt_dir is not None:
        start_epoch, start_global_step = load_ckpt(model_engine,scheduler,ckpt_dir,ckpt_tag)
    else:
        start_epoch = -1
        start_global_step = -1

    client_sd = {}

    for epoch in range(start_epoch+1,10):
        start_time = time.time()
        for global_step, batch in tqdm(enumerate(trainloader,start=start_global_step+1)):
            batch = batch.cuda()
            batch = batch.to(next(model_engine.parameters()).dtype) # data的dtype可能和模型不对应，比如开了bf16

            outputs = model_engine(batch)
            loss = outputs.mean()

            # loss = model_engine.train_step(batch)

            model_engine.backward(loss)
            model_engine.step()

        scheduler.step()

        #save checkpoint
        if epoch % save_interval == 0:
            client_sd['global_step'] = global_step
            client_sd['epoch'] = epoch
            client_sd['scheduler'] = scheduler.state_dict()
            ckpt_id = loss.item()
            with torch.no_grad():
                model_engine.save_checkpoint(save_dir=save_dir,tag=f'epoch_{epoch}',client_state = client_sd)

        if args.local_rank == 0:
            end_time = time.time()
            print(f'local_rank: {args.local_rank}',torch.cuda.max_memory_allocated()/1e9,'GB')
            print(f'local_rank: {args.local_rank}',end_time - start_time,'seconds')
        break
