from shared_encoder import SharedEncoder
import torch
from torchsummary import summary

if __name__ == "__main__":
    net = SharedEncoder(in_channels=1, hidden_dim=256).to("cpu")
    summary(net,(1,1200,80),device="cpu")