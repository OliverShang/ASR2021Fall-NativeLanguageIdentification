import torch.nn as nn
from torchsummary import summary
from shared_encoder import SharedEncoder
from accent_recognition_module import ARIntegration

# Input shape should be (batchsize,1,1200,40)

class SimpleAESRC(nn.Module):
    def __init__(self,classes=4):
        super(SimpleAESRC,self).__init__()
        self.encoder = SharedEncoder(in_channels=1,hidden_dim=256)
        self.lnk = nn.Linear(512,256)
        self.classifier = ARIntegration((114,256),classes)
    
    def forward(self,x):
        x = self.encoder(x)
        x = self.lnk(x)
        x = self.classifier(x)
        return x

model = SimpleAESRC().to("cpu")
summary(model,(1,1200,80),device="cpu")
