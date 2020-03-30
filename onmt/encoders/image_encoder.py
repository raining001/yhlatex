"""Image Encoder."""
import torch.nn as nn
import torch.nn.functional as F
import torch
from onmt.utils.self_atten import Self_Attn
from onmt.encoders.encoder import EncoderBase


class ImageEncoder(EncoderBase):
    """A simple encoder CNN -> RNN for image src.

    Args:
        num_layers (int): number of encoder layers.
        bidirectional (bool): bidirectional encoder.
        rnn_size (int): size of hidden states of the rnn.
        dropout (float): dropout probablity.

    """

    def __init__(self, num_layers, bidirectional, rnn_size, dropout, multi_scale,
                 image_chanel_size=3):
        super(ImageEncoder, self).__init__()
        self.multi_scale = multi_scale
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = rnn_size

        self.layer1 = nn.Conv2d(image_chanel_size, 64, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))

        self.layer2 = nn.Conv2d(64, 128, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer3 = nn.Conv2d(128, 256, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer4 = nn.Conv2d(256, 256, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer5 = nn.Conv2d(256, 512, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer6 = nn.Conv2d(512, 512, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))


        self.batch_norm1 = nn.BatchNorm2d(256)
        self.batch_norm2 = nn.BatchNorm2d(512)
        self.batch_norm3 = nn.BatchNorm2d(512)
        # self.layer_norm  = nn.LayerNorm(512)

        src_size = 512
        dropout = dropout[0] if type(dropout) is list else dropout
        # rnn_size 512
        # num_directions 2
        # rnn_size = 256

        self.rnn = nn.LSTM(src_size, int(src_size / self.num_directions),
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional)

        self.rnn2 = nn.LSTM(src_size, int(src_size / self.num_directions),
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional)
        # self.rnn2 = nn.LSTM(src_size, int(src_size / self.num_directions),
        #                    num_layers=num_layers,
        #                    dropout=dropout,
        #                    bidirectional=bidirectional)

        if self.multi_scale:
            src_size = 256
            self.rnn2 = nn.LSTM(src_size, int(src_size / self.num_directions),
                               num_layers=num_layers,
                               dropout=dropout,
                               bidirectional=bidirectional)

        # self.pos_lut = nn.Embedding(1000, src_size)

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
            opt.enc_rnn_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.multi_scale,
            image_channel_size
        )

    def load_pretrained_vectors(self, opt):
        """Pass in needed options only when modify function definition."""
        pass

    def forward(self, src, lengths=None):
        # print('src', src.size())
        """See :func:`onmt.encoders.encoder.EncoderBase.forward()`"""
        # (batch_size, 64, imgH, imgW)
        batch_size = src.size(0)
        # layer 1
        # print('输入图片',src.size())
        src = F.relu(self.layer1(src[:, :, :, :] - 0.5), True)
        # (batch_size, 64, imgH/2, imgW/2)
        src = F.max_pool2d(src, kernel_size=(2, 2), stride=(2, 2))

        # (batch_size, 128, imgH/2, imgW/2)
        # layer 2
        src = F.relu(self.layer2(src), True)
        # (batch_size, 128, imgH/2/2, imgW/2/2)
        src = F.max_pool2d(src, kernel_size=(2, 2), stride=(2, 2))
        #  (batch_size, 256, imgH/2/2, imgW/2/2)
        # layer 3
        # batch norm 1
        src = F.relu(self.batch_norm1(self.layer3(src)), True)
        # (batch_size, 256, imgH/2/2, imgW/2/2)
        # layer4
        src = F.relu(self.layer4(src), True)
        if self.multi_scale:
            self.highsrc = src

        # (batch_size, 256, imgH/2/2/2, imgW/2/2)
        src = F.max_pool2d(src, kernel_size=(1, 2), stride=(1, 2))
        # (batch_size, 512, imgH/2/2/2, imgW/2/2)
        # layer 5
        # batch norm 2
        src = F.relu(self.batch_norm2(self.layer5(src)), True)

        # (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
        src = F.max_pool2d(src, kernel_size=(2, 1), stride=(2, 1))
        # (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
        src = F.relu(self.batch_norm3(self.layer6(src)), True)
        # print('卷积后', src.size())
        # hidden_t = 0
        # # (batch_size, 512, H, W)
        # print('src', src.size())
        # out = src.view(src.size(2)*src.size(3), src.size(0), src.size(1))

        # 这里添加了位置信息，在每一行的开头会有一个用第几行数初始化，与原来每行的特征向量做个拼接之后在进行rowencoder
        # out, hidden_t = self.rowcol_origin(src)
        out, hidden_t = self.rowencoder(src)
        # print(('src1', src.size()))

        if self.multi_scale:
            # print('self.highsrc', self.highsrc.size())

            hightout, high_hidden_t = self.rowencoder_high(self.highsrc)

        # out, hidden_t = self.rowencoder(src)

        # out, hidden_t = self.origin_encoder(src, batch_size)

        # out += src.view(src.size(2)*src.size(3), src.size(0), src.size(1))
        # out = self.layer_norm(out) (layers*directions) x batch x dim.

        # out, hidden_t = self.colencoder(src)        #out (HxW, bz, 512) hidden_t (4, 20, 256) --(2个正向2个反向， bz, c/2)
        if self.multi_scale:
            # print('high_hidden_t', high_hidden_t[0].size(), 'hidden_t', hidden_t[0].size())
            h = torch.cat([hidden_t[0], high_hidden_t[0]], 2)
            c = torch.cat([hidden_t[1], high_hidden_t[1]], 2)
            hidden_t = (h, c)
            # print('hidden_t', hidden_t[0].size())

            return hidden_t, (out, hightout), lengths
        else:
            return hidden_t, out, lengths

    def rowcol_origin(self, src):
        all_outputs = []
        col_outputs = []
        for row in range(src.size(2)):
            inp = src[:, :, row, :].transpose(0, 2).transpose(1, 2)     # (bz, 512, W).(W, 512, bz).#(W, bz, 512)
            outputs, hidden_t = self.rnn(inp)
            all_outputs.append(outputs)
        out = torch.stack(all_outputs, 0)
        #
        for col in range(src.size(3)):
            inp = src[:, :, :, col].transpose(0, 2).transpose(1, 2)    # (bz, 512, H).(H, 512, bz).#(H, bz, 512)
            outputs, hidden_t2 = self.rnn2(inp)
            col_outputs.append(outputs)
        out_col = torch.stack(col_outputs, 1)

        out = out + out_col
        out = out.view(out.size(0) * out.size(1), out.size(2), out.size(3))
        return out, hidden_t2


    def colencoder(self, src):
        col_outputs = []
        for col in range(src.size(3)):
            inp = src[:, :, :, col].transpose(0, 2).transpose(1, 2)    # (bz, 512, H).(H, 512, bz).#(H, bz, 512)
            outputs, hidden_t2 = self.rnn2(inp)
            col_outputs.append(outputs)
        print('col_outputs', col_outputs[0].size())
        out = torch.cat(col_outputs, 0)
        return out, hidden_t2


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


    def origin_encoder(self, src, batch_size):
        all_outputs = []
        for row in range(src.size(2)):
            print('src', src.size())
            inp = src[:, :, row, :].transpose(0, 2) \
                .transpose(1, 2)
            row_vec = torch.Tensor(batch_size).type_as(inp.data) \
                .long().fill_(row)
            pos_emb = self.pos_lut(row_vec)
            with_pos = torch.cat(
                (pos_emb.view(1, pos_emb.size(0), pos_emb.size(1)), inp), 0)
            print('with_pos', with_pos.size())
            outputs, hidden_t = self.rnn(with_pos)
            all_outputs.append(outputs)
        out = torch.cat(all_outputs, 0)
        return out, hidden_t




    def rowencoder_high(self, src):
        print('src_high', src.size())
        all_outputs = []
        oldout = []
        for row in range(src.size(2)):
            inp = src[:, :, row, :].transpose(0, 2) \
                .transpose(1, 2)
            outputs, hidden_t = self.rnn2(inp)

            all_outputs.append(outputs)
        # print('all_outputs', all_outputs[0].size())
        out = torch.cat(all_outputs, 0)
        # print('out_high', out.size())
        return out, hidden_t
