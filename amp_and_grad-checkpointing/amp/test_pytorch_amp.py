import torch
import torch.nn as nn
import torch.optim as optim

from torch.cuda.amp import autocast, GradScaler

import time
import os

from eva_vit import create_eva_vit_g

# 定义训练函数
def train(epochs=10):
    torch.cuda.reset_peak_memory_stats()

    # 定义模型、优化器
    model = create_eva_vit_g(img_size=224,drop_path_rate=0.,use_checkpoint=False,precision="fp32").cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scaler = GradScaler()

    # 开始训练并记录时间和显存
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        inputs = torch.randn(16,3,224,224).cuda()
        with autocast():
            outputs = model(inputs)
            loss = outputs.sum()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    end_time = time.time()

    # 计算并返回时间和显存使用量
    duration = end_time - start_time
    memory = torch.cuda.max_memory_allocated()
    return duration, memory


duration, memory = train()
print(f"Opt Level: Pytorch(Equal to O1), Duration: {duration} seconds, Max Memory: {memory} bytes")
