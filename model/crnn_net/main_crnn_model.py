import torch.nn as nn
from torchsummary import summary
from model.crnn_net.shared_encoder import CRNNEncoder
from model.crnn_net.accent_recognition_module import CRNNClassifier
from utils.utils import CLASS_NUM, DEVICE
from model.rnn import BiGRU

# Input shape should be (batchsize,1,1200,40)

class CRNNet(nn.Module):
    def __init__(self,classes=CLASS_NUM,conv_output=224,hidden=128,out_feat=128,feature_only=False,
                ctc_branch=False,ctc_classes=27,ctc_len=353):
        super(CRNNet,self).__init__()
        self.encoder = CRNNEncoder(in_channels=1,hidden_dim=hidden)
        self.lnk = nn.Linear(hidden*2,hidden)
        self.actv = nn.Softmax(1)
        self.classifier = CRNNClassifier((conv_output,hidden),classes,hidden,out_feat,feature_only)
        self.classifier.train()
        # CTC
        self.enable_ctc = ctc_branch
        self.hidden_dim = hidden
        self.bi_gru = BiGRU(self.hidden_dim*2, ctc_len)
        self.ctcds1 = nn.Linear(ctc_len*2,self.hidden_dim,bias=False)
        self.ctcds1a = nn.Tanh()
        self.ctcds2 =nn.Linear(self.hidden_dim,ctc_classes,bias=False)
        self.ctcds2a = nn.LogSoftmax(1)
    
    def forward(self,x):
        y = self.encoder(x)
        x = self.lnk(y)
        x = self.classifier(x)
        if self.enable_ctc:
            y = self.bi_gru(y)
            y = self.ctcds1(y)
            y = self.ctcds1a(y)
            y = self.ctcds2(y)
            y = self.ctcds2a(y)
            y = y.permute(1,0,2) #[Time,Batch,Class]
            return x,y
        return x

# model = SimpleAESRC().to("cpu")
# summary(model,(1,1200,80),device="cpu")
