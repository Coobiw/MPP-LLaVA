from apex import amp

import torch.nn as nn
import torch

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(10, 20),nn.Linear(20,10))
    
    def forward(self,x):
        return self.model(x)

model = SimpleModel().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Initialization
opt_level = 'O2'
assert opt_level in ['O1','O2','O3'], 'ValueError'

print(f"model[0].weight.dtype: {model.model[0].weight.dtype}")
print(f"model[1].weight.dtype: {model.model[1].weight.dtype}")
model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
print(f"model[0].weight.dtype: {model.model[0].weight.dtype}")
print(f"model[1].weight.dtype: {model.model[1].weight.dtype}")

input_data = torch.randn(10, 10).cuda()
target = torch.randn(10, 10).cuda()


# Train your model
for epoch in range(1):
    print(f"input.dtype: {input_data.dtype}")
    output = model(input_data)
    print(f"output.dtype: {output.dtype}")
    loss = nn.MSELoss()(output, target)
    print(f"loss.dtype: {loss.dtype}")

    optimizer.zero_grad()
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    print(f"model[0].weight.grad.dtype: {model.model[0].weight.grad.dtype}")
    print(f"model[1].weight.grad.dtype: {model.model[1].weight.grad.dtype}")
    optimizer.step()

# Save checkpoint
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'amp': amp.state_dict()
}
# torch.save(checkpoint, 'amp_checkpoint.pt')


# Restore
# model = ...
# optimizer = ...
# checkpoint = torch.load('amp_checkpoint.pt')

# model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
# model.load_state_dict(checkpoint['model'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# amp.load_state_dict(checkpoint['amp'])
