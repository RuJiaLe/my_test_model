import torch
import torch.nn as nn
from .model_utils import Decoder_Part
from .model_utils import CLG, refine_module, frame_similarly
from .triplet_attention import TripletAttention
from .ssa import shunted_s, shunted_b


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

        # --------------------编码阶段--------------------

        self.encoder = Encoder_Model(pretrained=pretrained)

        # --------------------第四解码阶段--------------------

        self.frame_similarly_4 = frame_similarly()
        self.CLG4 = CLG(512, 512)
        self.decoder4 = Decoder_Part(512, 256)

        # --------------------第三解码阶段--------------------

        self.frame_similarly_3 = frame_similarly()
        self.CLG3 = CLG(256, 256)
        self.decoder3 = Decoder_Part(256, 128)

        # --------------------第二解码阶段--------------------

        self.frame_similarly_2 = frame_similarly()
        self.CLG2 = CLG(128, 128)
        self.decoder2 = Decoder_Part(128, 64)

        # --------------------第一解码阶段--------------------

        self.frame_similarly_1 = frame_similarly()
        self.CLG1 = CLG(64, 64)
        self.decoder1 = Decoder_Part(64, 32)

        # --------------------第0解码阶段--------------------

        self.frame_similarly_0 = frame_similarly()
        self.CLG0 = CLG(32, 32)
        self.decoder0 = Decoder_Part(32, 32)

        # --------------------refine--------------------

        # self.refine = refine_module()

        # --------------------output阶段--------------------

        self.out_CLG4 = CLG(256, 1)
        self.out_CLG3 = CLG(128, 1)
        self.out_CLG2 = CLG(64, 1)
        self.out_CLG1 = CLG(32, 1)
        self.out_CLG0 = CLG(32, 1)

        # --------------------上采样阶段--------------------

        self.Up_sample_32 = nn.Upsample(scale_factor=32, mode='bilinear')
        self.Up_sample_16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.Up_sample_8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.Up_sample_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.Up_sample_2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, frames):
        # --------------------编码阶段--------------------

        blocks_1 = []
        blocks_2 = []
        for i in range(len(frames)):
            block_1, block_2 = self.encoder(frames[i])
            blocks_1.append(block_1)
            blocks_2.append(block_2)

        # --------------------第四解码阶段--------------------
        in_x4 = []
        for i in range(len(frames)):
            t = blocks_1[i][3] + blocks_2[i][3]
            in_x4.append(t)

        in_x4 = self.frame_similarly_4(in_x4)

        x4 = []
        for i in range(len(frames)):
            x = self.decoder4(self.CLG4(in_x4[i]))
            x4.append(x)

        out4 = x4

        # --------------------第三解码阶段--------------------
        in_x3 = []
        for i in range(len(frames)):
            t = out4[i] + blocks_1[i][2] + blocks_2[i][2]
            in_x3.append(t)
        
        in_x3 = self.frame_similarly_3(in_x3)

        x3 = []
        for i in range(len(frames)):
            x = self.decoder3(self.CLG3(in_x3[i]))
            x3.append(x)

        out3 = x3

        # --------------------第二解码阶段--------------------
        in_x2 = []
        for i in range(len(frames)):
            t = out3[i] + blocks_1[i][1] + blocks_2[i][1]
            in_x2.append(t)
        
        in_x2 = self.frame_similarly_2(in_x2)

        x2 = []
        for i in range(len(frames)):
            x = self.decoder2(self.CLG2(in_x2[i]))
            x2.append(x)

        out2 = x2

        # --------------------第一解码阶段--------------------
        in_x1 = []
        for i in range(len(frames)):
            t = out2[i] + blocks_1[i][0] + blocks_2[i][0]
            in_x1.append(t)
        
        in_x1 = self.frame_similarly_1(in_x1)

        x1 = []
        for i in range(len(frames)):
            x = self.decoder1(self.CLG1(in_x1[i]))
            x1.append(x)

        out1 = x1

        # --------------------第0解码阶段--------------------

        in_x0 = []
        for i in range(len(frames)):
            t = out1[i]
            in_x0.append(t)

        in_x0 = self.frame_similarly_0(in_x0)
  
        x0 = []
        for i in range(len(frames)):
            x = self.decoder0(self.CLG0(in_x0[i]))
            x0.append(x)

        out0 = x0

        # --------------------输出阶段--------------------

        output4 = []
        for i in range(len(frames)):
            out = self.Up_sample_16(self.out_CLG4(out4[i]))
            output4.append(out)

        output3 = []
        for i in range(len(frames)):
            out = self.Up_sample_8(self.out_CLG3(out3[i]))
            output3.append(out)

        output2 = []
        for i in range(len(frames)):
            out = self.Up_sample_4(self.out_CLG2(out2[i]))
            output2.append(out)

        output1 = []
        for i in range(len(frames)):
            out = self.Up_sample_2(self.out_CLG1(out1[i]))
            output1.append(out)

        output0 = []
        for i in range(len(frames)):
            out = self.out_CLG0(out0[i])
            output0.append(out)

        # # --------------------refine--------------------

        # refine = []
        # for i in range(len(frames)):
        #     refine.append(
        #         self.refine(
        #             output0[i] + output1[i] + output2[i] + output3[i] +
        #             output4[i], blocks_2[i]))

        # refine_output = []
        # for i in range(len(refine[0])):  # 解码阶段
        #     stage = []
        #     for j in range(len(frames)):  # 帧数
        #         x = refine[j][i]
        #         stage.append(x)

        #     refine_output.append(stage)

        # return [output4, output3, output2, output1, output0], refine_output
        
        return [output4, output3, output2, output1, output0]