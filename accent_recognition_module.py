# The Accent Recognition Branch for the Module
# This is the PyTorch reimplementation of https://github.com/coolEphemeroptera/AESRC2020

import torch
import torch.nn as nn
from torchsummary import summary


class Utility:
    def __init__(self) -> None:
        pass

    @staticmethod
    def change_shape(x, dim: int, new_val: int):
        if isinstance(x,tuple):
            r = list(x)
        else:
            r = x.copy()
        r[dim] = new_val
        return r


class ARIntegration(nn.Module):
    """
    ARIntegration
    The shape of input should be a 3D vector [Batch size,Channels,Features]
    (Channel,Features) should be the output of Conv-RNN
    """
    def __init__(self, in_shape: list, accent_classes, hidden_dims=256,out_feat=128,feat_only=False):
        super(ARIntegration, self).__init__()

        # Parameters
        self.in_features = in_shape[1]  # [batch_size,in_shape]
        self.hidden_dims = hidden_dims

        # Layers
        self.dense1 = nn.Linear(in_features=in_shape[1], out_features=hidden_dims)
        self.dense1_actv = nn.Tanh()
        self.layernorm1 = nn.LayerNorm(Utility.change_shape(in_shape, 1, hidden_dims))
        # self.layernorm1 = nn.LayerNorm(256)
        # Integration (GAP)
        self.integration_gap = nn.AvgPool1d(hidden_dims)
        self.integration_gapflat = nn.Flatten()

        # Integration (BiGRU)
        self.integration_bigru = nn.GRU(input_size=hidden_dims, hidden_size=hidden_dims, batch_first=True,
                                        bidirectional=True)
        self.integration_bigruflat = nn.Flatten()

        # Layers
        self.bn1 = nn.BatchNorm1d(2 * hidden_dims)
        self.dense2 = nn.Linear(2 * hidden_dims, hidden_dims)
        self.bn2 = nn.BatchNorm1d(hidden_dims)

        # Classification Layers
        self.cl1 = nn.Linear(hidden_dims, out_feat)
        self.cl1a = nn.ReLU()
        self.cl2 = nn.Linear(out_feat, out_feat)
        self.cl2a = nn.ReLU()
        if not feat_only:
            self.fc = nn.Linear(out_feat, accent_classes)
            self.fca = nn.Softmax(1)
        else:
            self.fc = nn.Identity()
            self.fca = nn.Identity()


    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        x = self.dense1(x)
        x = self.dense1_actv(x)
        x = self.layernorm1(x)
        xp, _ = self.integration_bigru(x)
        x1, x2 = torch.chunk(xp, 2, dim=2)
        x = torch.cat((x1[:, -1, :], x2[:, 0, :]), 1)
        x = self.bn1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.cl1(x)
        x = self.cl1a(x)
        x = self.cl2(x)
        x = self.cl2a(x)
        x = self.fc(x)
        x = self.fca(x)
        return x


if __name__ == "__main__":
    model = ARIntegration([500, 256], 4).to("cpu")
    print(model)
    summary(model, input_size=(500, 256), device="cpu")
