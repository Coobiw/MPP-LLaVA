import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

model = nn.ModuleList([nn.Linear(10, 20),nn.Linear(20,10)]).cuda()

model[0].half() # 如果requires_grad为True，会有问题  ValueError: Attempting to unscale FP16 gradients.
# model[0].requires_grad_(False) # 这样的话，只有model[1]的requires_grad为True，且为float32


optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

scaler = GradScaler()

input_data = torch.randn(10, 10).cuda()
target = torch.randn(10, 10).cuda()

for epoch in range(1):
    optimizer.zero_grad()

    with autocast():  # autocast将自动管理必要的FP16到FP32的转换
        print(f"input.dtype: {input_data.dtype}")
        output = model[1](model[0](input_data))
        print(f"output.dtype: {output.dtype}")
        loss = nn.MSELoss()(output, target)
        print(f"loss.dtype: {loss.dtype}")
        print(f"model[0].weight.dtype: {model[0].weight.dtype}")
        print(f"model[1].weight.dtype: {model[1].weight.dtype}")

    scaler.scale(loss).backward()
    print(f"model[0].weight.grad.dtype: {model[0].weight.grad.dtype}")
    print(f"model[1].weight.grad.dtype: {model[1].weight.grad.dtype}")
    scaler.step(optimizer)
    scaler.update()
