import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(512, 512)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()


    def forward(self, x):
        x = checkpoint(self.layer1,x)
        x = self.relu1(x)
        x = checkpoint(self.layer2,x)
        return x

# 初始化模型和输入数据
model = SimpleModel().cuda()
# 需要设置requires_grad为True，因为所有linear层都被checkpoint了，而relu不会修改输出的requires_grad
# 这样就会导致输出的output的requires_grad为False，会出现大问题
# 如果forward里又有一个self.layer3,没被checkpoint也可以
# 或者forward里有一个最开始的self.layer1，没被checkpoint，同样也可以
inputs = torch.randn(int(1e6), 512,requires_grad=True).cuda() 
labels = torch.zeros(int(1e6),512).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters())

print(f"begin: {torch.cuda.memory_allocated()/1e9}")

model.train()

# 进行前向传播
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs,labels)
loss.backward()
optimizer.step()

print(f"final: {torch.cuda.memory_allocated()/1e9}")
