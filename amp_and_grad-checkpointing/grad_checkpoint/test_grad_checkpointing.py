import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from eva_vit import create_eva_vit_g

from torch.cuda.amp import autocast

def train(model):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    data = torch.randn(16,3,224,224).cuda()

    torch.cuda.reset_peak_memory_stats()

    with autocast():
        output = model(data)
    loss = output.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 训练后的显存使用情况
    final_memory = torch.cuda.max_memory_allocated()

    return final_memory/1e9

use_checkpoint = True
model = create_eva_vit_g(img_size=224,drop_path_rate=0.,use_checkpoint=use_checkpoint,precision="fp16").cuda()
print(f"参数量: {sum([param.numel() for param in model.parameters()])/1e9} B")

info = 'with' if use_checkpoint else 'without'
print(f"Memory used {info} checkpointing: ", train(model))

