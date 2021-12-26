import torch.nn as nn
from torchsummary import summary
from shared_encoder import SharedEncoder
from accent_recognition_module import ARIntegration
from utils import CLASS_NUM, DEVICE

# Input shape should be (batchsize,1,1200,40)

class SimpleAESRC(nn.Module):
    def __init__(self,classes=CLASS_NUM,conv_output=224,hidden=256,out_feat=128,feature_only=False):
        super(SimpleAESRC,self).__init__()
        self.encoder = SharedEncoder(in_channels=1,hidden_dim=hidden)
        self.lnk = nn.Linear(hidden*2,hidden)
        self.actv = nn.Softmax(1)
        self.classifier = ARIntegration((conv_output,hidden),classes,hidden,out_feat,feature_only)
        self.classifier.train()
    
    def forward(self,x):
        x = self.encoder(x)
        x = self.lnk(x)
        x = self.classifier(x)
        return x

# model = SimpleAESRC().to("cpu")
# summary(model,(1,1200,80),device="cpu")
