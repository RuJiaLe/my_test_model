import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .triplet_attention import TripletAttention
from .ssa import shunted_s


# Layer norm
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# De_block
class De_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(De_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), padding=1)
        self.ln = LayerNorm(in_channels, eps=1e-6, data_format='channels_first')

        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1))
        self.gelu = nn.GELU()

        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))

        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))

        self.attention1 = TripletAttention()

        self.Up_sample_2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        out = self.conv1(x)
        out = self.ln(out)

        out = self.conv2(out)
        out = self.gelu(out)

        out = self.conv3(out)

        out = out + self.conv1x1(x)

        out = self.Up_sample_2(out)

        out = self.attention1(out)

        return out


# CBS
class CBS(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(CBS, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)

        return x


# CBR
class CBR(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.conv(x)
        out = self.gelu(out)

        return out


# ASPP模块
class block_aspp_moudle(nn.Module):
    def __init__(self, in_dim, out_dim, output_stride=16, rates=[6, 12, 18]):
        super(block_aspp_moudle, self).__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []

        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=(1, 1), bias=False),
                          nn.ReLU(inplace=True))
        )

        # other rates
        for r in rates:
            self.features.append(
                nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=(3, 3),
                                        dilation=(r, r), padding=r, bias=False),
                              nn.ReLU(inplace=True)
                              )
            )

        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=(1, 1), bias=False),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Conv2d(in_channels=out_dim * 5, out_channels=out_dim, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:], mode='bilinear', align_corners=True)

        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)

        out = self.fuse(out)

        return out


# Video_Decoder_Part
class Video_Decoder_Part(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Video_Decoder_Part, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), padding=1)
        self.ln = LayerNorm(in_channels, eps=1e-6, data_format='channels_first')

        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
        self.gelu = nn.GELU()

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1))

        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))

        self.att = TripletAttention()

        self.Up_sample_2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        out = self.conv1(x)
        out = self.ln(out)

        out = self.conv2(out)
        out = self.gelu(out)

        out = self.conv3(out)

        x = self.conv1x1(x)

        out = out + x

        out = self.Up_sample_2(out)

        out = self.att(out)

        return out


class refine_module(nn.Module):
    def __init__(self):
        super(refine_module, self).__init__()
        self.ass_s = shunted_s(pretrained=True)

        self.att = TripletAttention()

        self.CBR3 = CBR(256, 256)
        self.CBR2 = CBR(128, 128)
        self.CBR1 = CBR(64, 64)
        self.CBR0 = CBR(32, 32)

        self.de_block4 = De_Block(512, 256)
        self.de_block3 = De_Block(256, 128)
        self.de_block2 = De_Block(128, 64)
        self.de_block1 = De_Block(64, 32)
        self.de_block0 = De_Block(32, 32)

        self.CBS4 = CBS(256, 1)
        self.CBS3 = CBS(128, 1)
        self.CBS2 = CBS(64, 1)
        self.CBS1 = CBS(32, 1)
        self.CBS0 = CBS(32, 1)

        self.Up_sample_16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.Up_sample_8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.Up_sample_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.Up_sample_2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, in_blocks):
        x = torch.cat((x, x, x), dim=1)

        blocks = self.ass_s(x)

        out_block = []
        for i in range(len(blocks)):
            block = self.att(blocks[i])
            out_block.append(block)

        decoder4 = self.de_block4(out_block[-1] + in_blocks[3])
        decoder3 = self.de_block3(self.CBR3(decoder4 + out_block[-2] + in_blocks[2]))
        decoder2 = self.de_block2(self.CBR2(decoder3 + in_blocks[1]))
        decoder1 = self.de_block1(self.CBR1(decoder2 + in_blocks[0]))
        decoder0 = self.de_block0(self.CBR0(decoder1))

        out4 = self.Up_sample_16(self.CBS4(decoder4))
        out3 = self.Up_sample_8(self.CBS3(decoder3))
        out2 = self.Up_sample_4(self.CBS2(decoder2))
        out1 = self.Up_sample_2(self.CBS1(decoder1))
        out0 = self.CBS0(decoder0)

        return out4, out3, out2, out1, out0
