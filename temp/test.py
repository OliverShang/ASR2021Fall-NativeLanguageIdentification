from model.crnn_net.shared_encoder import CRNNEncoder
import torch
from torchsummary import summary

if __name__ == "__main__":
    net = CRNNEncoder(in_channels=1, hidden_dim=256).to("cpu")
    summary(net,(1,1200,80),device="cpu")