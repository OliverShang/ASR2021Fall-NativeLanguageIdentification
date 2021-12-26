from torch import nn
from rnn import *
from resnet import ResNet
from utils import DEVICE


class SharedEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, resnet_type="res18", ):
        super(SharedEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.resnet = ResNet(resnet_type=resnet_type, in_channels=in_channels)
        self.dense = nn.Linear(512, self.hidden_dim).to(DEVICE)
        self.LN = nn.LayerNorm(hidden_dim)
        self.activation = nn.Tanh()
        self.bi_gru = BiGRU(256, self.hidden_dim).to(DEVICE)

    def forward(self, x):
        x = self.resnet(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.LN(x)
        x = self.bi_gru(x)
        layer_norm = nn.LayerNorm(x.size(2)).to(DEVICE)
        x = layer_norm(x)
        return x
