import torch.nn as nn
import torch.nn.functional as F
import torch
from onmt.utils.self_atten import Self_Attn
from onmt.encoders.encoder import EncoderBase
import torch.nn.functional as F

import math


class DenseNet(EncoderBase):
    def __init__(self, num_layers, bidirectional, growthRate, nDenseBlocks, reduction, dropout, rnn_size, bottleneck):
        super(DenseNet, self).__init__()

        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = rnn_size

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=7, padding=1, stride=2,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks[0], bottleneck)

        nChannels += nDenseBlocks[0]*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks[1], bottleneck)
        nChannels += nDenseBlocks[1]*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks[2], bottleneck)
        nChannels += nDenseBlocks[2]*growthRate
        #
        self.trans3 = ext_tran(nChannels, rnn_size)


        dropout = dropout[0] if type(dropout) is list else dropout
        # rnn_size 512
        # num_directions 2
        # rnn_size = 256
        #
        self.rnn = nn.LSTM(rnn_size, int(rnn_size / self.num_directions),
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)


    @classmethod
    def from_opt(cls, opt, embeddings=None):
        """Alternate constructor.
        enc_layers=2
        brnn=True
        enc_rnn_size=500
        dropout=[0.3]

        """

        if embeddings is not None:
            raise ValueError("Cannot use embeddings with ImageEncoder.")
        # why is the model_opt.__dict__ check necessary?
        if "image_channel_size" not in opt.__dict__:
            image_channel_size = 3
        else:
            image_channel_size = opt.image_channel_size
        return cls(
            opt.enc_layers,
            opt.brnn,
            opt.growthrate,
            opt.nDenseBlocks,
            opt.reduction,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.enc_rnn_size,
            opt.bottleneck
        )

    #
    def forward(self, src, lengths=None):
        # print('src', src.size())
        out = F.relu(self.conv1(src[:, :, :, :] - 0.5), True)
        # print('conv1', out.size())
        out = F.max_pool2d(out, kernel_size=(2, 2), stride=(2, 2))
        # print('out', out.size())
        out = self.trans1(self.dense1(out), kernel_size=(2, 1), stride=(2, 1))
        # print('dense1', out.size())
        out = self.trans2(self.dense2(out), kernel_size=(1, 2), stride=(1, 2))
        # print('dense2', out.size())
        out = self.trans3(self.dense3(out))
        # print('dense3', out.size())
        # print()
        # cnn： torch.Size([20, 512, 5, 25])
        out, hidden_t = self.rowencoder(out)
        # print('rowencoder', out.size())
        # print()
        return hidden_t, out, lengths

    def rowencoder(self, src):
        all_outputs = []
        oldout = []
        for row in range(src.size(2)):
            inp = src[:, :, row, :].transpose(0, 2) \
                .transpose(1, 2)
            outputs, hidden_t = self.rnn(inp)

            all_outputs.append(outputs)
        # print('all_outputs', all_outputs[0].size())
        out = torch.cat(all_outputs, 0)
        return out, hidden_t

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), True))
        out = self.conv2(F.relu(self.bn2(out), True))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), True))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x, kernel_size, stride):
        out = self.conv1(F.relu(self.bn1(x),True))
        out = F.avg_pool2d(out, kernel_size, stride)
        return out

class ext_tran(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(ext_tran, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), True))
        return out
