from shared_encoder import SharedEncoder
import torch

net = SharedEncoder(in_channels=3, hidden_dim=256)
i = torch.autograd.Variable(torch.randn(5, 3, 224, 224))
i2 = net(i)
print(i2.shape)