import torch

from .material import SSIM, IOU, S_Loss
import torch.nn as nn

# Loss
bce_loss = nn.BCELoss(size_average=True)
ssim_loss = SSIM(window_size=11, size_average=True)
iou_loss = IOU(size_average=True)
s_loss = S_Loss()


def Loss(predict, target):
    bce_out = bce_loss(predict, target)
    ssim_out = 1 - ssim_loss(predict, target)
    iou_out = iou_loss(predict, target)
    s_out = s_loss(predict, target)

    loss = bce_out + ssim_out + iou_out + s_out

    return loss, bce_out, ssim_out, iou_out, s_out


def multi_loss(out1, out2, gts):
    all_loss = []

    for j in range(len(gts)):

        if torch.cuda.is_available():
            frame_loss = torch.tensor(0.0).cuda()
        else:
            frame_loss = torch.tensor(0.0)

        for i in range(len(out1)):
            loss1 = Loss(out1[i][j], gts[j])
            loss2 = Loss(out2[i][j], gts[j])
            frame_loss += (loss1[0] + loss2[0])

        all_loss.append(frame_loss)

    return all_loss
