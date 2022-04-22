from torch.utils.data import Dataset
import os
import random
from PIL import Image
import numpy as np


class VideoDataset(Dataset):
    def __init__(self, root_dir="", train_set_list=None, training=True, transforms=None, clip_len=4):
        super(VideoDataset, self).__init__()

        self.root_dir = root_dir
        self.clip_len = clip_len
        self.training = training
        self.transforms = transforms
        self.train_set_list = train_set_list
        self.frames = []

        for train_set in self.train_set_list:
            video_root = os.path.join(root_dir, train_set).replace('\\', '/')
            sequence_list = sorted(os.listdir(video_root))

            for sequence in sequence_list:
                sequence_info = self.get_frame_list(train_set, sequence)
                self.frames += self.get_clips(sequence_info)

    def get_frame_list(self, train_set, sequence):
        image_path_root = os.path.join(self.root_dir, train_set, sequence, "Imgs").replace('\\', '/')
        frame_list = sorted(os.listdir(image_path_root))
        sequence_info = []

        for i in range(len(frame_list)):
            image_path = os.path.join(self.root_dir, train_set, sequence, "Imgs", frame_list[i]).replace('\\', '/')

            frame_name = frame_list[i].split('.')[0]
            gt_name = frame_name + '.png'
            gt_path = os.path.join(self.root_dir, train_set, sequence, "ground-truth", gt_name).replace('\\', '/')

            frame_info = {"image_path": image_path,
                          "gt_path": gt_path}

            sequence_info.append(frame_info)

        return sequence_info

    def get_clips(self, sequence_info):
        clips = []

        for i in range(int(len(sequence_info) / self.clip_len)):
            clips.append(sequence_info[self.clip_len * i: self.clip_len * (i + 1)])

        finish = self.clip_len * (int(len(sequence_info) / self.clip_len))

        if finish < len(sequence_info):
            clips.append(sequence_info[len(sequence_info) - self.clip_len: len(sequence_info)])

        return clips

    def get_frame(self, frame_info):
        image_path = frame_info["image_path"]
        image = Image.open(image_path).convert("RGB")
        gt_path = frame_info["gt_path"]
        gt = Image.open(gt_path).convert("L")

        sample = {"image": image, "gt": gt, "path": image_path}

        return sample

    def __getitem__(self, idx):
        frame = self.frames[idx]

        frame_output = []

        if self.training and random.randint(0, 1):
            frame = frame[::-1]

        for i in range(len(frame)):
            item = self.get_frame(frame[i])
            frame_output.append(item)

        frame_output = self.transforms(frame_output)

        return frame_output

    def __len__(self):
        return len(self.frames)


class ImageDataset(Dataset):
    def __init__(self, root_dir="", train_set_list=None, image_transform=None, clip_len=4):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.transforms = image_transform
        self.clip_len = clip_len
        self.lists = []
        self.train_set_list = train_set_list

        for train_set in self.train_set_list:
            image_root = os.path.join(root_dir, train_set).replace('\\', '/')
            image_info = self.get_image_list(image_root)
            self.lists += self.get_clips(image_info)

    def get_image_list(self, image_root):
        image_list = sorted(os.listdir(os.path.join(image_root, "Imgs").replace('\\', '/')))
        image_info = []

        for i in range(len(image_list)):
            image_name = image_list[i].split('.')[0]
            gt_name = image_name + '.png'
            info = {"image_path": os.path.join(image_root, "Imgs", image_list[i]).replace('\\', '/'),
                    "gt_path": os.path.join(image_root, "ground-truth", gt_name).replace('\\', '/')}

            image_info.append(info)

        return image_info

    def get_clips(self, image_info):
        clips = []

        for i in range(int(len(image_info))):
            image_info_list = []
            for _ in range(self.clip_len):
                image_info_list.append(image_info[i])

            clips.append(image_info_list)

        return clips

    def motion_simulate(self, image_output):  # [{}, {}, {}, {}]
        output = []

        flag = random.randint(-1, 1)
        if flag == -1:
            # 向左平移
            if random.randint(0, 1):
                for i in range(len(image_output)):
                    img, gt = image_output[i]['image'], image_output[i]['gt']

                    img = img.transform(img.size, Image.AFFINE, (1, 0, i * 10, 0, 1, 0))
                    gt = gt.transform(gt.size, Image.AFFINE, (1, 0, i * 10, 0, 1, 0))

                    image_output[i]['image'], image_output[i]['gt'] = img, gt
            # 向右平移
            else:
                for i in range(len(image_output)):
                    img, gt = image_output[i]['image'], image_output[i]['gt']

                    img = img.transform(img.size, Image.AFFINE, (1, 0, -i * 10, 0, 1, 0))
                    gt = gt.transform(gt.size, Image.AFFINE, (1, 0, -i * 10, 0, 1, 0))

                    image_output[i]['image'], image_output[i]['gt'] = img, gt

        elif flag == 0:
            # 向上平移
            if random.randint(0, 1):
                for i in range(len(image_output)):
                    img, gt = image_output[i]['image'], image_output[i]['gt']

                    img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, i * 10))
                    gt = gt.transform(gt.size, Image.AFFINE, (1, 0, 0, 0, 1, i * 10))

                    image_output[i]['image'], image_output[i]['gt'] = img, gt
            # 向下平移
            else:
                for i in range(len(image_output)):
                    img, gt = image_output[i]['image'], image_output[i]['gt']

                    img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, -i * 10))
                    gt = gt.transform(gt.size, Image.AFFINE, (1, 0, 0, 0, 1, -i * 10))

                    image_output[i]['image'], image_output[i]['gt'] = img, gt
        else:
            # 逆时针旋转
            if random.randint(0, 1):
                for i in range(len(image_output)):
                    img, gt = image_output[i]['image'], image_output[i]['gt']

                    img = img.rotate(i * 5, Image.BILINEAR)
                    gt = gt.rotate(i * 5, Image.BILINEAR)

                    image_output[i]['image'], image_output[i]['gt'] = img, gt
            # 顺时针旋转
            else:
                for i in range(len(image_output)):
                    img, gt = image_output[i]['image'], image_output[i]['gt']

                    img = img.rotate(-i * 5, Image.BILINEAR)
                    gt = gt.rotate(-i * 5, Image.BILINEAR)

                    image_output[i]['image'], image_output[i]['gt'] = img, gt

        return image_output

    def get_image(self, image_info):
        image_path = image_info["image_path"]
        image = Image.open(image_path).convert("RGB")
        gt_path = image_info["gt_path"]
        gt = Image.open(gt_path).convert("L")

        sample = {"image": image, "gt": gt, "path": image_path}

        return sample

    def __getitem__(self, idx):
        images = self.lists[idx]

        image_output = []

        for i in range(len(images)):
            item = self.get_image(images[i])
            image_output.append(item)

        # 变换
        image_output = self.motion_simulate(image_output)

        if random.randint(0, 1):
            image_output = image_output[::-1]

        frame_output = self.transforms(image_output)

        return frame_output

    def __len__(self):
        return len(self.lists)


# Eval_data load
class EvalDataset(Dataset):
    def __init__(self, root_dir="./predict_data/", train_set_list=None, training=True, transforms=None, clip_len=4):
        super(EvalDataset, self).__init__()

        self.root_dir = root_dir
        self.clip_len = clip_len
        self.training = training
        self.transforms = transforms
        self.frames = []
        self.train_set_list = train_set_list

        for train_set in self.train_set_list:
            video_root = os.path.join(root_dir, train_set).replace('\\', '/')
            sequence_list = sorted(os.listdir(video_root))

            for sequence in sequence_list:
                sequence_info = self.get_frame_list(train_set, sequence)
                self.frames += self.get_clips(sequence_info)

    def get_frame_list(self, train_set, sequence):
        predict_path_root = os.path.join(self.root_dir, train_set, sequence, "predicts").replace('\\', '/')
        frame_list = sorted(os.listdir(predict_path_root))
        sequence_info = []

        for i in range(len(frame_list)):
            predict_path = os.path.join(self.root_dir, train_set, sequence, "predicts", frame_list[i]).replace('\\', '/')
            gt_path = os.path.join(self.root_dir, train_set, sequence, "ground-truth", frame_list[i]).replace('\\', '/')

            frame_info = {"predict_path": predict_path,
                          "gt_path": gt_path}

            sequence_info.append(frame_info)

        return sequence_info

    def get_clips(self, sequence_info):
        clips = []

        for i in range(int(len(sequence_info) / self.clip_len)):
            clips.append(sequence_info[self.clip_len * i: self.clip_len * (i + 1)])

        finish = self.clip_len * (int(len(sequence_info) / self.clip_len))

        if finish < len(sequence_info):
            clips.append(sequence_info[len(sequence_info) - self.clip_len: len(sequence_info)])

        return clips

    def get_frame(self, frame_info):
        predict_path = frame_info["predict_path"]

        predict = Image.open(predict_path).convert("L")
        gt_path = frame_info["gt_path"]
        gt = Image.open(gt_path).convert("L")

        sample = {"predict": predict, "gt": gt}

        return sample

    def __getitem__(self, idx):
        frame = self.frames[idx]

        frame_output = []
        if self.training and random.randint(0, 1):
            frame = frame[::-1]

        for i in range(len(frame)):
            item = self.get_frame(frame[i])
            frame_output.append(item)

        frame_output = self.transforms(frame_output)

        return frame_output

    def __len__(self):
        return len(self.frames)
