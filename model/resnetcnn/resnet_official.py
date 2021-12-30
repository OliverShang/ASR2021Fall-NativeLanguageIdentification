import torch
import torch.nn as nn
from torch.nn.modules import activation
import torchvision.models as models 
from torchsummary import summary

class ResNetCNN(nn.Module):
    def __init__(self,classes=2) -> None:
        super(ResNetCNN,self).__init__()
        self.resnet = models.resnet18()
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(1000,classes)
        self.actv = nn.Softmax(1)
        self.resnet.train()

    def forward(self,x):
        x = torch.concat([x,x,x],dim=1)
        x = self.resnet(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.actv(x)
        return x

if __name__ == "__main__":
    pass