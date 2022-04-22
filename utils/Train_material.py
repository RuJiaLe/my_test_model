import os
import torch
import time
from .material import Eval_mae, Eval_F_measure, Eval_S_measure
from datetime import datetime
from .Loss import multi_loss
import torch.optim as optim
from .material import adjust_lr, Log_image
from tqdm import tqdm
import time
from .config import args
from torch.utils import tensorboard
from itertools import cycle

# best_val_loss
best_val_loss = 0.0


# val
def val(packs, model):
    MAES, S_measures = 0.0, 0.0
    avg_p, avg_r = 0.0, 0.0

    model.eval()
    with torch.no_grad():
        images, gts = [], []
        for pack in packs:
            image, gt = pack["image"], pack["gt"]

            if torch.cuda.is_available():
                image, gt = image.cuda(), gt.cuda()

            images.append(image)
            gts.append(gt)

        # 解码
        out1, out2 = model(images)

        # Loss 计算
        predicts = out2[-1]

        for predict_, gt_ in (zip(predicts, gts)):
            for k in range(predict_.size()[0]):
                predict = predict_[k, :, :, :].unsqueeze(0)
                gt = gt_[k, :, :, :].unsqueeze(0)

                mae = Eval_mae(predict, gt)
                MAES += mae.data

                prec, recall = Eval_F_measure(predict, gt)
                avg_p += prec.data
                avg_r += recall.data

                S_measure = Eval_S_measure(predict, gt)
                S_measures += S_measure.data

        return MAES / args.batch_size, S_measures / args.batch_size


# 开始训练
def train(train_data, val_data, model, optimizer, Epoch, writer):
    model.train()
    model.freeze_bn()

    total_step = len(train_data)
    all_losses = 0.0
    already_time = 0.0

    MAES = 0.0
    S_measures = 0.0

    for i, packs in enumerate(zip(train_data, cycle(val_data))):
        start_time = time.time()
        i = i + 1
        optimizer.zero_grad()

        images, gts = [], []
        for pack in packs[0]:
            image, gt = pack["image"], pack["gt"]

            if torch.cuda.is_available():
                image, gt = image.cuda(), gt.cuda()

            images.append(image)
            gts.append(gt)

        # 解码
        out1, out2 = model(images)  # 第4阶段, 第3阶段, 第2阶段, 第1阶段, 第0阶段

        if i % 100 == 0:
            Log_image(out1, out2, gts, writer, i // 100)

        # Loss 计算
        loss = multi_loss(out1, out2, gts)
        for k in range(args.clip_len):
            all_losses += loss[k].data

        # 反向传播
        torch.autograd.backward(loss)
        optimizer.step()

        # 验证
        mae, s_measure = val(packs[1], model)
        MAES += mae
        S_measures += s_measure

        end_time = time.time()
        speed = end_time - start_time
        already_time += speed

        # 显示与记录内容
        if i % 10 == 0 or i == total_step:
            print('{},Epoch:{:02d}/{:02d},Step:{:0.2f}%|{:04d}/{:04d},Loss:[train:{:0.4f}|mae:{:0.4f}|s_val:{:0.4f}],time:{:0.2f}/{:0.2f}min'.
                  format(datetime.now().strftime('%m/%d/%H:%M'),
                         Epoch, args.total_epoch, (i / total_step) * 100, i,
                         total_step, all_losses / (i * args.clip_len), MAES / (i * args.clip_len), S_measures / (i * args.clip_len),
                         already_time / 60.0, (total_step - i) * speed / 60.0))

            writer.add_scalar(tag='Loss/all_loss', scalar_value=all_losses / (i * args.clip_len), global_step=i)

            writer.add_scalar(tag='Val/mae', scalar_value=MAES / (i * args.clip_len), global_step=i)
            writer.add_scalar(tag='Val/s_val', scalar_value=S_measures / (i * args.clip_len), global_step=i)


    # 验证与模型保存
    global best_val_loss
    if Epoch % 1 == 0:
        val_loss = 1 - MAES / (total_step * args.clip_len) + S_measures / (total_step * args.clip_len)

        writer.add_scalar(tag='Best_model_loss', scalar_value=val_loss, global_step=Epoch)

        if best_val_loss < val_loss:  # big is best
            best_val_loss = val_loss

            torch.save({'epoch': Epoch, 'state_dict': model.state_dict(), 'best_val_loss': best_val_loss, 'optimizer': optimizer.state_dict()},
                       args.model_path + '/video_best_model.pth.tar')

            print('this is best_model_Epoch: {}'.format(Epoch))

    # 保存最后一次模
    torch.save({'epoch': Epoch, 'state_dict': model.state_dict(), 'best_val_loss': best_val_loss, 'optimizer': optimizer.state_dict()},
               args.model_path + '/video_last_model.pth.tar')


# 模型加载模式
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def start_train(train_dataloader, val_dataloader, model):
    # tensorboard
    writer = tensorboard.SummaryWriter('./Log/train')

    # 加载至cuda
    if torch.cuda.is_available():
        model.cuda()

    # 优化器
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 加载模型
    start_epoch = args.start_epoch
    global best_val_loss

    path = args.model_path + '/image_best_model.pth.tar'
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=torch.device(device))
        model.load_state_dict(checkpoint["state_dict"])

        print('Load pre_train model Done !!! ')

    path = args.model_path + '/video_last_model.pth.tar'
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=torch.device(device))
        model.load_state_dict(checkpoint["state_dict"])

        start_epoch = checkpoint['epoch']
        start_epoch = start_epoch + 1
        best_val_loss = checkpoint['best_val_loss']
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        print('Load video_last_model Done !!! ')

    for epoch in range(start_epoch, args.total_epoch + 1):
        adjust_lr(optimizer, epoch, args.decay_rate, args.decay_epoch)

        train(train_dataloader, val_dataloader, model, optimizer, epoch, writer)

    writer.close()
