import numpy as np
import os
from PIL import Image
import torch
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F
import torch.nn as nn
import random


# ******************************计算模型参数******************************
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


# ******************************学习率衰减******************************
def adjust_lr(optimizer, epoch, decay_rate=0.9, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    # print("lr=", optimizer.param_groups[0]["lr"])

    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


# ******************************图片保存******************************
def Save_result(img, frame_image_path, save_path):
    path_split = frame_image_path.split("/")[3:]

    image_save_path = os.path.join(save_path, path_split[0], path_split[1], "predicts")

    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

    image_save_path = image_save_path + '/' + path_split[3][:-4] + '.png'

    img = img.detach().cpu().numpy().squeeze()

    img = (img - img.min()) / (img.max() - img.min() + 1e-20)
    img = img * 255.0
    img = img.astype(np.uint8)

    img[img >= 128] = 255
    img[img < 128] = 0

    img = Image.fromarray(img)
    img.save(image_save_path)


# ******************************Loss******************************
# IOU Loss
def _iou(predict, target, size_average=True):
    b = predict.shape[0]
    IoU = 0.0
    for i in range(0, b):
        # compute the IoU of the foreground
        Iand1 = torch.sum(target[i, :, :, :] * predict[i, :, :, :])
        Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(predict[i, :, :, :]) - Iand1
        IoU1 = Iand1 / Ior1

        # IoU loss is (1-IoU1)
        IoU = IoU + (1 - IoU1)

    return IoU / b


class IOU(torch.nn.Module):
    def __init__(self, size_average=True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, predict, target):
        return _iou(predict, target, self.size_average)


# SSIM Loss
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


# S_loss
class S_Loss(nn.Module):
    def __init__(self):
        super(S_Loss, self).__init__()

    def forward(self, x, label):
        loss = F.smooth_l1_loss(x, label)
        return loss


# ******************************Eval******************************
# RP
def _eval_pr(y_pred, y, num):
    if torch.cuda.is_available():
        prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
        thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
    else:
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
    return prec, recall


# eval_e
def _eval_e(y_pred, y, num):
    if torch.cuda.is_available():
        score = torch.zeros(num).cuda()
    else:
        score = torch.zeros(num)

    for i in range(num):
        fm = y_pred - y_pred.mean()
        gt = y - y.mean()
        align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
        enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
        score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)
    return score.max()


# object
def _object(pred, gt):
    temp = pred[gt == 1]
    x = temp.mean()
    sigma_x = temp.std()
    score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
    return score


# S_object
def _S_object(pred, gt):
    fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
    bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
    o_fg = _object(fg, gt)
    o_bg = _object(bg, 1 - gt)
    u = gt.mean()
    Q = u * o_fg + (1 - u) * o_bg
    return Q


# _centroid
def _centroid(gt):
    rows, cols = gt.size()[-2:]
    gt = gt.view(rows, cols)
    if gt.sum() == 0:
        if torch.cuda.is_available():
            X = torch.eye(1).cuda() * round(cols / 2)
            Y = torch.eye(1).cuda() * round(rows / 2)
        else:
            X = torch.eye(1) * round(cols / 2)
            Y = torch.eye(1) * round(rows / 2)
    else:
        total = gt.sum()
        if torch.cuda.is_available():
            i = torch.from_numpy(np.arange(0, cols)).cuda().float()
            j = torch.from_numpy(np.arange(0, rows)).cuda().float()
        else:
            i = torch.from_numpy(np.arange(0, cols)).float()
            j = torch.from_numpy(np.arange(0, rows)).float()
        X = torch.round((gt.sum(dim=0) * i).sum() / total)
        Y = torch.round((gt.sum(dim=1) * j).sum() / total)
    return X.long(), Y.long()


# _divideGT
def _divideGT(gt, X, Y):
    h, w = gt.size()[-2:]
    area = h * w
    gt = gt.view(h, w)
    LT = gt[:Y, :X]
    RT = gt[:Y, X:w]
    LB = gt[Y:h, :X]
    RB = gt[Y:h, X:w]
    X = X.float()
    Y = Y.float()
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1 - w1 - w2 - w3
    return LT, RT, LB, RB, w1, w2, w3, w4


# _dividePrediction
def _dividePrediction(pred, X, Y):
    h, w = pred.size()[-2:]
    pred = pred.view(h, w)
    LT = pred[:Y, :X]
    RT = pred[:Y, X:w]
    LB = pred[Y:h, :X]
    RB = pred[Y:h, X:w]
    return LT, RT, LB, RB


# _ssim
def S_ssim(pred, gt):
    gt = gt.float()
    h, w = pred.size()[-2:]
    N = h * w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
    sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

    aplha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

    if aplha != 0:
        Q = aplha / (beta + 1e-20)
    elif aplha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0
    return Q


# S_region
def _S_region(pred, gt):
    X, Y = _centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = _divideGT(gt, X, Y)
    p1, p2, p3, p4 = _dividePrediction(pred, X, Y)
    Q1 = S_ssim(p1, gt1)
    Q2 = S_ssim(p2, gt2)
    Q3 = S_ssim(p3, gt3)
    Q4 = S_ssim(p4, gt4)
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
    # print(Q)
    return Q


# MAE
def Eval_mae(pred, gt):
    with torch.no_grad():
        mae = torch.abs(pred - gt).mean()
        return mae


# Max F-measure
def Eval_F_measure(pred, gt):
    with torch.no_grad():
        prec, recall = _eval_pr(pred, gt, 255)
        return prec, recall


# E-measure
def Eval_E_measure(pred, gt):
    max_e = 0.0
    with torch.no_grad():
        max_e = _eval_e(pred, gt, 255)
        return max_e


# S-measure
def Eval_S_measure(pred, gt):
    alpha, avg_q = 0.5, 0.0
    with torch.no_grad():
        y = gt.mean()
        if y == 0:
            x = pred.mean()
            Q = 1.0 - x
        elif y == 1:
            x = pred.mean()
            Q = x
        else:
            Q = alpha * _S_object(pred, gt) + (1 - alpha) * _S_region(pred, gt)
            if Q.item() < 0:
                # Q = torch.FLoatTensor([0.0])
                Q = torch.tensor([0.0])

        avg_q = Q
        if avg_q == avg_q:
            return avg_q
        else:
            return 0.0


# ******************************Log******************************
# Log_image
def Log_image(out1, out2, gt, writer, step):  # out[阶段][帧]         [帧][阶段]

    output1_4 = torch.cat(out1[0], dim=0)
    output1_3 = torch.cat(out1[1], dim=0)
    output1_2 = torch.cat(out1[2], dim=0)
    output1_1 = torch.cat(out1[3], dim=0)
    output1_0 = torch.cat(out1[4], dim=0)

    output2_4 = torch.cat(out2[0], dim=0)
    output2_3 = torch.cat(out2[1], dim=0)
    output2_2 = torch.cat(out2[2], dim=0)
    output2_1 = torch.cat(out2[3], dim=0)
    output2_0 = torch.cat(out2[0], dim=0)

    gts = torch.cat(gt, dim=0)

    writer.add_images(tag='Image/output2_4', img_tensor=output2_4, global_step=step, dataformats='NCHW')
    writer.add_images(tag='Image/output2_3', img_tensor=output2_3, global_step=step, dataformats='NCHW')
    writer.add_images(tag='Image/output2_2', img_tensor=output2_2, global_step=step, dataformats='NCHW')
    writer.add_images(tag='Image/output2_1', img_tensor=output2_1, global_step=step, dataformats='NCHW')
    writer.add_images(tag='Image/output2_0', img_tensor=output2_0, global_step=step, dataformats='NCHW')

    writer.add_images(tag='Image/output1_4', img_tensor=output1_4, global_step=step, dataformats='NCHW')
    writer.add_images(tag='Image/output1_3', img_tensor=output1_3, global_step=step, dataformats='NCHW')
    writer.add_images(tag='Image/output1_2', img_tensor=output1_2, global_step=step, dataformats='NCHW')
    writer.add_images(tag='Image/output1_1', img_tensor=output1_1, global_step=step, dataformats='NCHW')
    writer.add_images(tag='Image/output1_0', img_tensor=output1_0, global_step=step, dataformats='NCHW')

    writer.add_images(tag='Image/GT', img_tensor=gts, global_step=step, dataformats='NCHW')
