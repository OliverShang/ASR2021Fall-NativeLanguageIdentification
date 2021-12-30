from torch import nn
from model.rnn import *
from model.resnet import ResNet
from utils.utils import DEVICE


class CRNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, resnet_type="res18", ):
        super(CRNNEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.resnet = ResNet(resnet_type=resnet_type, in_channels=in_channels)
        self.dense = nn.Linear(512, self.hidden_dim).to(DEVICE)
        self.LN = nn.LayerNorm(hidden_dim)
        self.activation = nn.Tanh()
        self.bi_gru = BiGRU(self.hidden_dim, self.hidden_dim).to(DEVICE)
        #self.ln = nn.LayerNorm(x.size(2))

    def forward(self, x):
        x = self.resnet(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.LN(x)
        x = self.bi_gru(x)
        #x = layer_norm(x)
        return x
