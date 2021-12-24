from torch import nn
from rnn import *
from resnet import ResNet
from utils import DEVICE


class SharedEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, resnet_type="res34", ):
        super(SharedEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.resnet = ResNet(resnet_type=resnet_type, in_channels=in_channels)
        self.LN = nn.LayerNorm(hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.resnet(x)
        dense = nn.Linear(x.size(2), self.hidden_dim).to(DEVICE)
        x = dense(x)
        x = self.activation(x)
        x = self.LN(x)

        bi_gru = BiGRU(x.size(2), self.hidden_dim).to(DEVICE)
        x = bi_gru(x)
        layer_norm = nn.LayerNorm(x.size(2)).to(DEVICE)
        x = layer_norm(x)
        return x
