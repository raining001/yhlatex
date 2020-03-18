"""Image Encoder."""
import torch.nn as nn
import torch.nn.functional as F
import torch
from onmt.utils.self_atten import Self_Attn
from onmt.encoders.encoder import EncoderBase


class SAM_Encoder(EncoderBase):
    """
    Spatial alignment module encoder
    """

    def __init__(self, num_layers, bidirectional, rnn_size, dropout, multi_scale,
                 image_chanel_size=3):
        super(SAM_Encoder, self).__init__()
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
        self.sam = SAM(maxT=160, depth=6, in_channels=512, num_channels=128)

        self.rnn = nn.LSTM(src_size, int(src_size / self.num_directions),
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional)

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

        s_atten = self.sam(src)
        s_atten = s_atten.view(s_atten.size(2)*s_atten.size(3), s_atten.size(0), s_atten.size(1))
        out, hidden_t = self.rowencoder(src)

        return hidden_t, (out, s_atten), lengths


    def rowencoder(self, src):
        all_outputs = []

        for row in range(src.size(2)):
            inp = src[:, :, row, :].transpose(0, 2) \
                .transpose(1, 2)
            outputs, hidden_t = self.rnn(inp)

            all_outputs.append(outputs)
        out = torch.cat(all_outputs, 0)
        return out, hidden_t



class SAM(nn.Module):
    def __init__(self,  maxT, depth, in_channels, num_channels):
        super(SAM, self).__init__()
        strides = []
        conv_ksizes = []
        deconv_ksizes = []
        # conv
        strides = [(1,2) ,(2,1), (1,1)]
        for i in range(0, int(depth / 2)):
            conv_ksizes.append((3, 3))
            deconv_ksizes.append((3,3))
        padding = [(1,1),(1,1),(0,0)]
        convs = [nn.Sequential(nn.Conv2d(in_channels, num_channels,
                                        conv_ksizes[0],
                                        strides[0],
                                        padding[0]),
                               nn.BatchNorm2d(num_channels),
                               nn.ReLU(True))]
        for i in range(1, int(depth / 2)):
            convs.append(nn.Sequential(nn.Conv2d(num_channels, num_channels,
                                                tuple(conv_ksizes[i]),
                                                tuple(strides[i]),
                                                padding[i]),
                                       nn.BatchNorm2d(num_channels),
                                       nn.ReLU(True)))
        self.convs = nn.Sequential(*convs)
        # deconvs
        deconvs = []
        self.deconv1 = []


        self.deconv0 = nn.ConvTranspose2d(num_channels, num_channels,
                                          kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)).cuda()
        self.deconv1 = nn.ConvTranspose2d(num_channels, num_channels,
                                          kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)).cuda()
        self.deconv2 = nn.ConvTranspose2d(num_channels, maxT,
                                          kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)).cuda()


        self.deconv0_bn = nn.BatchNorm2d(num_channels)
        self.deconv1_bn = nn.BatchNorm2d(num_channels)


    def forward(self, x):
        src = x
        conv_feats = []
        xsize = []
        # 三个下采样，大小[bz, C, W, H] --> [bz, C, W, H/2] --> [bz, C, W/2, H/2] --> [bz, C, W/2-2, H/2-2]
        for i in range(0, len(self.convs)):
            xsize.append(x.size())
            conv_feats.append(x)
            x = self.convs[i](x)
        x = F.relu(self.deconv0_bn(self.deconv0(x, output_size=xsize[2])), True)
        # [bz, C, W / 2, H / 2]
        x = x + conv_feats[2]
        x = F.relu(self.deconv1_bn(self.deconv1(x, output_size=xsize[1])), True)
        #  [bz, C, W, H/2]
        x = x + conv_feats[1]
        attn = F.softmax(self.deconv2(x, output_size=xsize[0]), 1)

        return attn


