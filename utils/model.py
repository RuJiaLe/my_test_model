import torch
import torch.nn as nn
from .model_utils import Video_Decoder_Part
from .model_utils import CBS, CBR, refine_module
from .ConvGRU import ConvGRUCell
from .triplet_attention import TripletAttention
from .ssa import shunted_s


class Encoder_Model(nn.Module):
    def __init__(self, pretrained=True):
        super(Encoder_Model, self).__init__()

        self.ssa_s_1 = shunted_s(pretrained=pretrained)
        self.ssa_s_2 = shunted_s(pretrained=pretrained)

        self.attention_module_1_1 = TripletAttention()
        self.attention_module_2_1 = TripletAttention()
        self.attention_module_3_1 = TripletAttention()
        self.attention_module_4_1 = TripletAttention()

        self.attention_module_1_2 = TripletAttention()
        self.attention_module_2_2 = TripletAttention()
        self.attention_module_3_2 = TripletAttention()
        self.attention_module_4_2 = TripletAttention()

    def forward(self, x):  # (1, 3, 256, 256)
        blocks_1 = self.ssa_s_1(x)
        blocks_2 = self.ssa_s_2(x)

        blocks_1[0] = self.attention_module_1_1(blocks_1[0])
        blocks_1[1] = self.attention_module_2_1(blocks_1[1])
        blocks_1[2] = self.attention_module_3_1(blocks_1[2])
        blocks_1[3] = self.attention_module_4_1(blocks_1[3])

        blocks_2[0] = self.attention_module_1_2(blocks_2[0])
        blocks_2[1] = self.attention_module_2_2(blocks_2[1])
        blocks_2[2] = self.attention_module_3_2(blocks_2[2])
        blocks_2[3] = self.attention_module_4_2(blocks_2[3])

        return blocks_1, blocks_2


class Model(nn.Module):
    def __init__(self, pretrained=True):
        super(Model, self).__init__()
        if pretrained:
            for key in self.state_dict():
                if 'resnet' not in key:
                    self.init_layer(key)
        # --------------------编码阶段--------------------
        self.encoder = Encoder_Model(pretrained=pretrained)

        # --------------------第四解码阶段--------------------

        self.decoder4 = Video_Decoder_Part(512, 256)

        # --------------------第三解码阶段--------------------

        self.CBR3 = CBR(256, 256)

        self.decoder3 = Video_Decoder_Part(256, 128)

        # --------------------第二解码阶段--------------------
        self.CBR2 = CBR(128, 128)

        self.decoder2 = Video_Decoder_Part(128, 64)

        # self.ConvGRU2 = ConvGRUCell(64, 64)

        # --------------------第一解码阶段--------------------
        self.CBR1 = CBR(64, 64)

        self.decoder1 = Video_Decoder_Part(64, 32)

        # self.ConvGRU1 = ConvGRUCell(32, 32)

        # --------------------第0解码阶段--------------------
        self.CBR0 = CBR(32, 32)
        self.decoder0 = Video_Decoder_Part(32, 32)

        # --------------------refine--------------------
        self.refine = refine_module()
        # self.ConvGRU0 = ConvGRUCell(1, 1)
        # --------------------output阶段--------------------
        self.CBS4 = CBS(256, 1)
        self.CBS3 = CBS(128, 1)
        self.CBS2 = CBS(64, 1)
        self.CBS1 = CBS(32, 1)
        self.CBS0 = CBS(32, 1)

        # --------------------上采样阶段--------------------
        self.Up_sample_32 = nn.Upsample(scale_factor=32, mode='bilinear')
        self.Up_sample_16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.Up_sample_8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.Up_sample_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.Up_sample_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.Up_sample_1_2 = nn.Upsample(scale_factor=0.5, mode='bilinear')

    def forward(self, frames):  # (1, 2048, 8, 8)  [block1, block2, block3, block4]
        # --------------------编码阶段--------------------
        blocks_1 = []
        blocks_2 = []
        for i in range(len(frames)):
            block_1, block_2 = self.encoder(frames[i])
            blocks_1.append(block_1)
            blocks_2.append(block_2)

        # --------------------第四解码阶段--------------------
        x4 = []
        for i in range(len(frames)):
            x = self.decoder4(blocks_1[i][3])
            x4.append(x)

        out4 = x4

        # --------------------第三解码阶段--------------------
        x3 = []

        for i in range(len(frames)):
            x = out4[i] + blocks_1[i][2]
            x = self.CBR3(x)
            x = self.decoder3(x)
            x3.append(x)

        out3 = x3

        # --------------------第二解码阶段--------------------
        x2 = []

        for i in range(len(frames)):
            x = out3[i] + blocks_1[i][1]
            x = self.CBR2(x)
            x = self.decoder2(x)
            x2.append(x)

        # out2 = [None]
        # for i in range(len(frames)):
        #     out = self.ConvGRU2(x2[i], out2[i])
        #     out2.append(out)

        # out2 = out2[1:]
        out2 = x2

        # --------------------第一解码阶段--------------------
        x1 = []
        sum_1 = out2[-1]
        for i in range(len(frames) - 1):
            sum_1 += out2[i]

        for i in range(len(frames)):
            x = out2[i] + blocks_1[i][0]
            x = self.CBR1(x + sum_1)
            x = self.decoder1(x)
            x1.append(x)

        # out1 = [None]
        # for i in range(len(frames)):
        #     out = self.ConvGRU1(x1[i], out1[i])
        #     out1.append(out)

        # out1 = out1[1:]
        out1 = x1

        # --------------------第一解码阶段--------------------
        x0 = []
        sum_0 = out1[-1]
        for i in range(len(frames) - 1):
            sum_0 += out1[i]

        for i in range(len(frames)):
            x = out1[i]
            x = self.CBR0(x + sum_0)
            x = self.decoder0(x)
            x0.append(x)
        
        out0 = x0

        # --------------------输出阶段--------------------
        output4 = []
        for i in range(len(frames)):
            out = self.Up_sample_16(self.CBS4(out4[i]))
            output4.append(out)

        output3 = []
        for i in range(len(frames)):
            out = self.Up_sample_8(self.CBS3(out3[i]))
            output3.append(out)

        output2 = []
        for i in range(len(frames)):
            out = self.Up_sample_4(self.CBS2(out2[i]))
            output2.append(out)

        output1 = []
        for i in range(len(frames)):
            out = self.Up_sample_2(self.CBS1(out1[i]))
            output1.append(out)
        
        output0 = []
        for i in range(len(frames)):
            out = self.CBS0(out0[i])
            output0.append(out)

        # --------------------refine--------------------
        refine = []
        for i in range(len(frames)):
            refine.append(self.refine(output0[i] + output1[i] + output2[i] + output3[i] + output4[i], blocks_2[i]))

        refine_output = []
        for i in range(len(refine[0])):  # 解码阶段
            stage = []
            for j in range(len(frames)):  # 帧数
                x = refine[j][i]
                stage.append(x)

            refine_output.append(stage)

        return [output4, output3, output2, output1, output0], refine_output
