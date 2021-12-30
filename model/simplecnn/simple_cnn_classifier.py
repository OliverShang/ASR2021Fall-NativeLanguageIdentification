# A simple accent classifier based on CNN

import torch
import torch.nn as nn
from torchsummary import summary


class SimpleARCNN(nn.Module):
    """
    SimpleARCNN
    The input should be a 4D tensor [Batch size,1 (Channel),MFCC Features,Frequency Components]
    """
    def __init__(self, accent_classes=2,conv_out=1280,fc_ne=128,feature_only=False):
        super(SimpleARCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (3, 3), (1, 1), padding=(1, 1))
        self.actv1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d((3, 3), (2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 32, (3, 3), (1, 1), padding=(1, 1))
        self.actv2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d((3, 3), (2, 2), padding=(1, 1))
        self.conv3 = nn.Conv2d(32, 64, (3, 3), (1, 1), padding=(1, 1))
        self.actv3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d((3, 3), (2, 2), padding=(1, 1))
        self.conv4 = nn.Conv2d(64, 128, (3, 5), (1, 1), padding=(1, 1))
        self.actv4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d((3, 5), (1, 5), padding=(1, 2))
        self.conv5 = nn.Conv2d(128, 256, (3, 5), (1, 1), padding=(1, 1))
        self.actv5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d((3, 5), (1, 5), padding=(1, 2))
        self.pool6 = nn.AvgPool2d((5, 5), (5, 5))
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(conv_out, fc_ne)
        self.dropout1 = nn.Dropout(0.2)
        if not feature_only:
            self.fc = nn.Linear(fc_ne, accent_classes)
            self.fca = nn.Softmax(1)
        else:
            self.fc = nn.Identity()
            self.fca = nn.Identity()

    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        x = self.conv1(x)
        x = self.actv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.actv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.actv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.actv4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.actv5(x)
        x = self.pool5(x)
        x = self.pool6(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.fc(x)
        x = self.fca(x)
        return x

class TinyCNN(nn.Module):
    """
    SimpleARCNN
    The input should be a 4D tensor [Batch size,1 (Channel),MFCC Features,Frequency Components]
    """
    def __init__(self, accent_classes=2,conv_output=1280,hidden=128,feature_only=False):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), (1, 1), padding=(1, 1))
        self.actv1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d((3, 3), (2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, (3, 3), (1, 1), padding=(1, 1))
        self.actv2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d((3, 3), (2, 2), padding=(1, 1))
        self.dropout1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(conv_output, hidden)
        self.dropout2 = nn.Dropout(0.5)
        if not feature_only:
            self.fc = nn.Linear(hidden, accent_classes)
            self.fca = nn.Softmax(1)
        else:
            self.fc = nn.Identity()
            self.fca = nn.Identity()

    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        x = self.conv1(x)
        x = self.actv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.actv2(x)
        x = self.pool2(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.fc(x)
        x = self.fca(x)
        return x

if __name__ == "__main__":
    model = TinyCNN(accent_classes=4, conv_output=80640).to("cuda")
    print(model)
    summary(model, input_size=(1, 39, 501), device="cuda")
