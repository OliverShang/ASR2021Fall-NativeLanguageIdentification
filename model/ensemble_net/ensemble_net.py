# Models
from model.crnn_net.main_crnn_model import CRNNet
from model.simplecnn.simple_cnn_classifier import SimpleARCNN, TinyCNN
from model.crnn_net.shared_encoder import CRNNEncoder
from torchsummary import summary
import torch.nn as nn
from utils.utils import CLASS_NUM, DEVICE
from model.crnn_net.accent_recognition_module import CRNNClassifier
from model.rnn import BiGRU
import torch

class EnsembleNet1(nn.Module):
    def __init__(self,classes=4,conv_output=70,hidden=256,out_feat=128,feature_only=False):
        super(EnsembleNet1,self).__init__()
        self.encoder = CRNNEncoder(in_channels=1,hidden_dim=hidden)
        self.lnk = nn.Linear(hidden*2,hidden)
        self.bigru = BiGRU(hidden, hidden)
        self.dense1 = nn.Linear(2 * hidden, hidden)
        self.dense2 = nn.Linear(hidden, classes)
        self.actv = nn.Softmax(1)
        self.classifier = CRNNClassifier((conv_output,hidden),classes,hidden,out_feat,False)
        self.classifier.train()
    
    def forward(self,x):
        x = self.encoder(x)
        x = self.lnk(x)
        y = self.bigru(x)
        y1, y2 = torch.chunk(y, 2, dim=2)
        y = torch.cat((y1[:, -1, :], y2[:, 0, :]), 1)
        y = self.dense1(y)
        y = self.dense2(y)
        x = self.classifier(x)
        y = self.actv(y)
        return 0.5 * x + 0.5 * y

class EnsembleNet2(nn.Module):
    def __init__(self,classes=8,conv_output1=256,conv_output2=256,hidden=256,out_feat=128,feature_only=False):
        super(EnsembleNet2,self).__init__()
        self.cnn = SimpleARCNN(classes, conv_out=conv_output1, fc_ne=1024, feature_only=False)
        self.crnn = CRNNet(classes, conv_output=conv_output2, hidden=hidden, out_feat=out_feat, feature_only=False)
    
    def forward(self,x):
        x1 = self.cnn(x)
        x2 = self.crnn(x)
        return 0.8 * x1 + 0.2 * x2


if __name__ == '__main__':
    model = EnsembleNet1().to("cuda")
    print(model)
    summary(model, input_size=(1, 45, 1103), device="cuda")