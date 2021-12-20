import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.basic = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.shortcut = shortcut

    def forward(self, x):
        out = self.basic(x)
        residual = x if self.shortcut is None else self.shortcut(x)  # 计算残差
        out += residual
        return nn.ReLU(inplace=True)(out)


# ResNet
class ResNet(nn.Module):
    def __init__(self, in_channels, resnet_type="res34"):
        super(ResNet, self).__init__()
        self.layers = []
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        if resnet_type == "res18":
            self.body = self.make_layers([2, 2, 2, 2])
        elif resnet_type == "res34":
            self.body = self.make_layers([3, 4, 6, 3])
        elif resnet_type == "res50":
            self.body = self.make_layers([3, 4, 6, 3])
        elif resnet_type == "res101":
            self.body = self.make_layers([3, 4, 23, 3])
        elif resnet_type == "res152":
            self.body = self.make_layers([3, 8, 36, 3])

    def make_layers(self, block_list):
        for index, block_num in enumerate(block_list):
            if index != 0:
                shortcut = nn.Sequential(
                    nn.Conv2d(64 * 2 ** (index - 1), 64 * 2 ** index, 1, 2, bias=False),
                    nn.BatchNorm2d(64 * 2 ** index)
                )
                self.layers.append(ResidualBlock(64 * 2 ** (index - 1), 64 * 2 ** index, 2, shortcut))  # 每次变化通道数时进行下采样
            for _ in range(0 if index == 0 else 1, block_num):
                self.layers.append(ResidualBlock(64 * 2 ** index, 64 * 2 ** index, 1))
        return nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.body(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)
        return x


