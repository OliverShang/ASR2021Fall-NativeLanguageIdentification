import torch.nn as nn
import torch


class BiGRU(nn.Module):
    def __init__(self, n_input, n_hidden,return_hn=False,batch_first=True):
        super(BiGRU, self).__init__()
        self.rhn = return_hn
        self.gru = nn.GRU(n_input, n_hidden, batch_first=batch_first, bidirectional=True)

    def forward(self, x):
        gru_output, h_n = self.gru(x)
        # concat the last hidden state from two direction
        # output = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), 1)
        # output = self.linear(hidden_out)
        # output = self.linear(gru_output[:, -1, :])
        if self.rhn:
            return gru_output, h_n
        return gru_output

class BiLSTM(nn.Module):
    def __init__(self, n_input, n_hidden,return_hn=False,batch_first=True):
        super(BiLSTM, self).__init__()
        self.rhn = return_hn
        self.gru = nn.LSTM(n_input, n_hidden, batch_first=batch_first, bidirectional=True)

    def forward(self, x):
        gru_output, h_n = self.gru(x)
        # concat the last hidden state from two direction
        # output = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), 1)
        # output = self.linear(hidden_out)
        # output = self.linear(gru_output[:, -1, :])
        if self.rhn:
            return gru_output, h_n
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
