"""Image Encoder."""
import torch.nn as nn
import torch.nn.functional as F
import torch
from onmt.utils.self_atten import Self_Attn
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.transformer import TransformerEncoder

class TransformerEncoder(EncoderBase):
    """A simple encoder CNN -> RNN for image src.

    Args:
        num_layers (int): number of encoder layers.
        bidirectional (bool): bidirectional encoder.
        rnn_size (int): size of hidden states of the rnn.
        dropout (float): dropout probablity.

    """

    def __init__(self, num_layers, bidirectional, rnn_size, dropout, multi_scale,
                 image_chanel_size=3):
        super(TransformerEncoder, self).__init__()
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

