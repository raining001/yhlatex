"""Image Encoder."""
import torch.nn as nn
import torch.nn.functional as F
import torch
from onmt.utils.self_atten import Self_Attn
from onmt.encoders.encoder import EncoderBase
import math


class SAM_Encoder(EncoderBase):
    """
    Spatial alignment module encoder
    """

    def __init__(self, num_layers, bidirectional, rnn_size, dropout, multi_scale,
                 image_chanel_size=3):
        super(SAM_Encoder, self).__init__()

        # self.resnet = resnet45(compress_layer=True)
        # self.vgg = VGG(num_layers, bidirectional, rnn_size, dropout, multi_scale,
        #          image_chanel_size)
        # self.sam = SAM(maxT=160, depth=8)
        self.FPN = FPN(Bottleneck)
        src_size = 512
        self.rnn = nn.LSTM(src_size, int(src_size / 2),
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
        # features = self.resnet(src)
        # features = self.vgg(src)

        # s_atten = self.sam(features)
        features = self.FPN(src)
        memory, hidden_t = self.rowencoder(features)


        # print('features', features[-1].size())
        # print('memory', memory.size())

        # print('hidden_t', hidden_t[0].size(), hidden_t[1].size())
        # hidden_t = (torch.zeros(4, src.size(0), 256).type_as(s_atten.data), torch.zeros(4, src.size(0), 256).type_as(s_atten.data))
        return hidden_t, memory, None
        # return hidden_t, (features[-1], s_atten), None

    def rowencoder(self, src):
        all_outputs = []
        for row in range(src.size(2)):
            inp = src[:, :, row, :].transpose(0, 2) \
                .transpose(1, 2)
            outputs, hidden_t = self.rnn(inp)
            # print('outputs', outputs.size())
            all_outputs.append(outputs)             # outputs torch.Size([W, bz, c])

        out = torch.cat(all_outputs, 0)
        # print('out', out.size())                    # out torch.Size([WxH, bz, c])

        return out, hidden_t



class SAM(nn.Module):
    def __init__(self,  maxT, depth, in_channels=512, num_channels=128):
        super(SAM, self).__init__()
        conv_ksizes = []
        deconv_ksizes = []
        # conv
        strides = [(2,2), (1,2) ,(2,1), (1,1)]

        # strides = [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)]
        channels = [(32,64), (64,128), (128,256), (256,512)]
        fpn_stripe = [(1, 1), (2, 2), (1, 1), (1, 1)]
        fpn = []

        for i in range(4):
            fpn.append(nn.Sequential(nn.Conv2d(channels[i][0], channels[i][1],
                                              (3, 3),
                                              fpn_stripe[i],
                                              1, 1),
                                     nn.BatchNorm2d(channels[i][1]),
                                     nn.ReLU(True)))
        self.fpn = nn.Sequential(*fpn)


        depth = 8
        for i in range(0, int(depth / 2)):
            conv_ksizes.append((3, 3))
            deconv_ksizes.append((3,3))
        padding = [(1,1),(1,1),(1,1),(0,0)]

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
        self.deconv2 = nn.ConvTranspose2d(num_channels, num_channels,
                                          kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)).cuda()
        self.deconv3 = nn.ConvTranspose2d(num_channels, maxT,
                                          kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)).cuda()


        self.deconv0_bn = nn.BatchNorm2d(num_channels)
        self.deconv1_bn = nn.BatchNorm2d(num_channels)
        self.deconv2_bn = nn.BatchNorm2d(num_channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        x = input[0]
        for i in range(0, len(self.fpn)):
            x = self.fpn[i](x) + input[i+1]

        conv_feats = []
        xsize = []
        for i in range(0, len(self.convs)):
            xsize.append(x.size())
            conv_feats.append(x)
            x = self.convs[i](x)
        x = F.relu(self.deconv0_bn(self.deconv0(x, output_size=xsize[3])), True)
        x = x + conv_feats[3]
        x = F.relu(self.deconv1_bn(self.deconv1(x, output_size=xsize[2])), True)
        x = x + conv_feats[2]
        x = F.relu(self.deconv2_bn(self.deconv2(x, output_size=xsize[1])), True)
        x = x + conv_feats[1]

        # attn = F.softmax(self.deconv2(x, output_size=xsize[0]), 1)
        attn = F.sigmoid(self.deconv3(x, output_size=xsize[0]))

        return attn





def conv1x1(in_planes,out_planes,stride=1):
    return nn.Conv2d(in_planes,out_planes,kernel_size =1,stride =stride,bias=False)
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out







class BasicBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    # model = ResNet(BasicBlock, [3, 4, 6, 6, 3], strides, compress_layer)
    # strides = [(1, 1), (1, 1), (2, 2), (2, 2), (2, 2), (2, 1)]
    def __init__(self, block, compress_layer=True):
        self.inplanes = 64
        strides = [(2, 2), (2, 2), (1, 1), (1, 2), (2, 1), (1, 1)]
        layers = [3, 4, 6, 6, 3]
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=strides[0], padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[2])
        self.layer3 = self._make_layer(block, 128, layers[2], stride=strides[3])
        self.layer4 = self._make_layer(block, 256, layers[3], stride=strides[4])
        self.layer5 = self._make_layer(block, 512, layers[4], stride=strides[5])

        self.compress_layer = compress_layer
        if compress_layer:
            # for handwritten
            self.layer6 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=(3, 1), padding=(0, 0), stride=(1, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace = True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, multiscale = False):
        # print('src', x.size())
        # (batch_size, 64, imgH/2, imgW/2)
        out_features = []
        x = F.relu(self.bn1(self.conv1(x)), True)
        # x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        # print('x0', x.size())

        x = self.layer1(x)
        out_features.append(x)
        # print('x1', x.size())

        x = self.layer2(x)
        out_features.append(x)
        # print('x2', x.size())

        x = self.layer3(x)
        out_features.append(x)
        # print('x3', x.size())

        x = self.layer4(x)
        out_features.append(x)
        # print('x4', x.size())

        x = self.layer5(x)
        # print('x5', x.size())

        out_features.append(x)
        return out_features




class FPN(nn.Module):
    def __init__(self, block):
        super(FPN, self).__init__()
        self.inplanes = 64
        strides = [(1, 1), (2, 2), (2, 2), (2, 2), (2, 2)]
        layers = [2, 2, 2, 2]
        super(FPN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=strides[0], padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        # self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        # self.smooth3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != block.expansion * planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, block.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(block.expansion * planes)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # Bottom-up
        print("x", x.size())

        x = F.relu(self.bn1(self.conv1(x)), True)
        c1 = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)        #

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))

        p3 = self._upsample_add(p4, self.latlayer2(c3))

        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        # p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        # p2 = self.smooth3(p2)
        print("p3", p3.size())
        return p3


def resnet45(compress_layer):
    model = ResNet(BasicBlock)
    return model



class VGG(nn.Module):

    def __init__(self, num_layers, bidirectional, rnn_size, dropout, multi_scale,
                 image_chanel_size=3):
        super(VGG, self).__init__()
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

        self.toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        # self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.smooth2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.smooth3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        self.batch_norm1 = nn.BatchNorm2d(256)
        self.batch_norm2 = nn.BatchNorm2d(512)
        self.batch_norm3 = nn.BatchNorm2d(512)


    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        """See :func:`onmt.encoders.encoder.EncoderBase.forward()`"""
        # (batch_size, 64, imgH, imgW)
        # layer 1
        # print('输入图片',src.size())
        c1 = F.relu(self.layer1(x), True)
        # (batch_size, 64, imgH/2, imgW/2)
        c1 = F.max_pool2d(c1, kernel_size=(2, 2), stride=(2, 2))

        # (batch_size, 128, imgH/2, imgW/2)
        # layer 2
        c2 = F.relu(self.layer2(c1), True)
        # (batch_size, 128, imgH/2/2, imgW/2/2)
        c2 = F.max_pool2d(c2, kernel_size=(2, 2), stride=(2, 2))
        #  (batch_size, 256, imgH/2/2, imgW/2/2)
        # layer 3
        # batch norm 1
        c3 = F.relu(self.batch_norm1(self.layer3(c2)), True)
        # (batch_size, 256, imgH/2/2, imgW/2/2)
        # layer4
        c4 = F.relu(self.layer4(c3), True)

        # (batch_size, 256, imgH/2/2/2, imgW/2/2)
        c4 = F.max_pool2d(c4, kernel_size=(1, 2), stride=(1, 2))
        # (batch_size, 512, imgH/2/2/2, imgW/2/2)
        # layer 5
        # batch norm 2
        c5 = F.relu(self.batch_norm2(self.layer5(c4)), True)

        # (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
        c5 = F.max_pool2d(c5, kernel_size=(2, 1), stride=(2, 1))
        # (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
        c6 = F.relu(self.batch_norm3(self.layer6(c5)), True)


        p6 = self.toplayer(c6)
        p5 = self._upsample_add(p6, self.latlayer1(c5))

        p4 = self._upsample_add(p5, self.latlayer2(c4))

        p3 = self._upsample_add(p4, self.latlayer3(c3))

        # p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        # p5 = self.smooth1(p5)
        # p4 = self.smooth2(p4)
        p3 = self.smooth3(p3)

        return p3


