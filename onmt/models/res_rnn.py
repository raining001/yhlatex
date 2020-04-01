import torch
import torch.nn as nn


class ResLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(ResLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        self.batch_norm = nn.BatchNorm1d(512)
        for i in range(num_layers):
            self.layers.append(nn.LSTM(input_size, rnn_size,
                           num_layers=1,
                           dropout=0.3,
                           bidirectional=True).cuda())


        # self.rnn_ = nn.LSTM(input_size, rnn_size,
        #                     num_layers=2,
        #                     dropout=0,
        #                     bidirectional=True).cuda()
    def forward(self, src):

        h = []
        c = []
        x, hidden_0 = self.layers[0](src)
        h.append(hidden_0[0])
        x = self.dropout(x)
        c.append(hidden_0[1])
        outputs, hidden_1 = self.layers[1](x, hidden_0)

        h.append(hidden_1[0])
        c.append(hidden_1[1])
        hidden = (torch.cat(h, 0), torch.cat(c,0))
        outputs = outputs + x
        outputs = outputs.transpose(1,0).transpose(2,1)
        outputs = self.batch_norm(outputs)
        outputs = outputs.transpose(2,1).transpose(1,0)

        return outputs, hidden

    def tmp(self, input_feed, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input_feed, (h_0[i], c_0[i]))
            input_feed = h_1_i
            if i + 1 != self.num_layers:
                input_feed = self.dropout(input_feed)
                residual = input_feed

            h_1 += [h_1_i]

            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input_feed+residual, (h_1, c_1)

