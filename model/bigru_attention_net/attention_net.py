import torch.nn as nn
import torch
import numpy as np
from model.rnn import *
from torchsummary import summary

class BiGruAttentionNetBlock(nn.Module):
    def __init__(self,in_channel=32,out_channel=64) -> None:
        super(BiGruAttentionNetBlock,self).__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,(3,3))
        self.bn = nn.BatchNorm2d(out_channel)
        self.actv = nn.ReLU()
        self.pool = nn.MaxPool2d((2,2))
    
    def forward(self,x):
        x = x.type(torch.cuda.FloatTensor)
        x = self.conv(x)
        x = self.bn(x)
        x = self.actv(x)
        x = self.pool(x)
        return x


class BiGruAttentionNet(nn.Module):
    def __init__(self,classes=2) -> None:
        super(BiGruAttentionNet,self).__init__()
        self.out = 32
        self.hidden = 64
        self.blk1 = BiGruAttentionNetBlock(1,16)
        self.blk2 = BiGruAttentionNetBlock(16,self.out)
        self.rnnbfactv = nn.Tanh()
        self.rnn = BiGRU(self.out,self.hidden,return_hn=True,batch_first=False)
        self.attn = nn.Sequential(
            nn.Linear(self.hidden,self.hidden),
            nn.ReLU(inplace=True)
        )
        self.actv = nn.Softmax(-1)
        self.fc = nn.Linear(self.hidden,classes)
        self.fcactv = nn.Softmax(1)
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(1,16,(3,3))
        self.bn1= nn.BatchNorm2d(16)
        self.actv1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(16,self.out,(3,3))
        self.bn2 = nn.BatchNorm2d(self.out)
        self.actv2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d((2,2))

    def forward(self,x):
        x = x.type(torch.cuda.FloatTensor)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.actv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.actv2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.rnnbfactv(x)
        x = x.permute(0, 2, 1)
        x = x.permute(1, 0, 2)
        out,h_n = self.rnn(x)
        (fw,bw) = torch.chunk(out,2,dim=2)
        out = fw+bw
        out = out.permute(1,0,2)
        h_n = h_n.permute(1,0,2)
        h_n = torch.sum(h_n,dim=1)
        h_n = h_n.squeeze(dim=1)
        attention_w = self.attn(h_n)
        attention_w = attention_w.unsqueeze(dim=1)
        attn_ctx = torch.bmm(attention_w,out.transpose(1,2))
        softmaxw = self.actv(attn_ctx)
        x = torch.bmm(softmaxw,out)
        x = x.squeeze(1)
        x = self.fc(x)
        x = self.fcactv(x)
        return x
        