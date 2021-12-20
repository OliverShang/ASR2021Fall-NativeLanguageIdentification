import torch.nn as nn
import torch


class BiGRU(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(BiGRU, self).__init__()

        self.gru = nn.GRU(n_input, n_hidden, batch_first=True, bidirectional=True)
        # self.linear = nn.Linear(2 * n_hidden, n_class)

    def forward(self, x):
        gru_output, h_n = self.gru(x)
        # concat the last hidden state from two direction
        # output = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), 1)
        # output = self.linear(hidden_out)
        # output = self.linear(gru_output[:, -1, :])
        return gru_output


class GRU(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(GRU, self).__init__()

        self.gru = nn.GRU(n_input, n_hidden, batch_first=True, bidirectional=False)
        # self.linear = nn.Linear(2 * n_hidden, n_class)

    def forward(self, x):
        gru_output, h_n = self.gru(x)
        # concat the last hidden state from two direction
        # output = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), 1)
        # output = self.linear(hidden_out)
        # output = self.linear(gru_output[:, -1, :])
        return gru_output
