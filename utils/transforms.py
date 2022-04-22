import random
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


def get_Eval_transforms(input_size):
    return transforms.Compose([
        Eval_Resize(input_size),
        Eval_ToTensor()
    ])


def get_transforms(input_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return transforms.Compose([Resize(input_size),
                               ToTensor(),
                               Normalize(mean=mean, std=std)])


def get_train_transforms(input_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return transforms.Compose([Resize(input_size),
                               RandomFlip(),
                               Random_crop(15),
                               ToTensor(),
                               Normalize(mean=mean, std=std)])


class RandomFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    """

    def __call__(self, samples):
        rand_flip_index = random.randint(-1, 2)

        if rand_flip_index == 0:
            for i in range(len(samples)):
                sample = samples[i]
                image, gt = sample['image'], sample['gt']
                image = F.hflip(image)
                gt = F.hflip(gt)
                sample['image'], sample['gt'] = image, gt
                samples[i] = sample

        elif rand_flip_index == 1:
            for i in range(len(samples)):
                sample = samples[i]
                image, gt = sample['image'], sample['gt']
                image = F.vflip(image)
                gt = F.vflip(gt)
                sample['image'], sample['gt'] = image, gt
                samples[i] = sample

        else:
            for i in range(len(samples)):
                sample = samples[i]
                image, gt = sample['image'], sample['gt']
                image = F.vflip(F.hflip(image))
                gt = F.vflip(F.hflip(gt))
                sample['image'], sample['gt'] = image, gt
                samples[i] = sample

        return samples


class Resize(object):
    """ Resize PIL image use both for training and inference"""

    def __init__(self, size):
        self.size = size

    def __call__(self, samples):
        for i in range(len(samples)):
            sample = samples[i]
            image, gt = sample['image'], sample['gt']

            image = F.resize(image, self.size, InterpolationMode.BILINEAR)
            gt = F.resize(gt, self.size, InterpolationMode.NEAREST)

            sample['image'], sample['gt'] = image, gt
            samples[i] = sample

        return samples


class Eval_Resize(object):
    """ Resize PIL image use both for training and inference"""

    def __init__(self, size):
        self.size = size

    def __call__(self, samples):
        for i in range(len(samples)):
            sample = samples[i]
            predict, gt = sample['predict'], sample['gt']

            predict = F.resize(predict, self.size, InterpolationMode.BILINEAR)
            gt = F.resize(gt, self.size, InterpolationMode.NEAREST)

            sample['predict'], sample['gt'] = predict, gt
            samples[i] = sample

        return samples


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, samples):
        for i in range(len(samples)):
            sample = samples[i]
            image, gt = sample['image'], sample['gt']

            image = F.to_tensor(image)

            gt = F.to_tensor(gt)

            sample['image'], sample['gt'] = image, gt
            samples[i] = sample

        return samples


class Eval_ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, samples):
        for i in range(len(samples)):
            sample = samples[i]
            predict, gt = sample['predict'], sample['gt']

            predict = F.to_tensor(predict)
            gt = F.to_tensor(gt)

            sample['predict'], sample['gt'] = predict, gt
            samples[i] = sample

        return samples


class Normalize(object):
    """ Normalize a tensor image with mean and standard deviation.
        args:    tensor (Tensor) ? Tensor image of size (C, H, W) to be normalized.
        Returns: Normalized Tensor image.
    """

    # default caffe mode
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, samples):
        for i in range(len(samples)):
            sample = samples[i]

            image = sample['image']
            image = F.normalize(image, self.mean, self.std)

            sample["image"] = image

            samples[i] = sample

        return samples


class Random_crop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, samples):
        x, y = random.randint(0, self.crop_size), random.randint(0, self.crop_size)

        width, height = samples[0]["image"].size
        assert samples[0]["image"].size == samples[0]["gt"].size
        region = [x, y, width - x, height - y]

        for i in range(len(samples)):
            sample = samples[i]
            image, gt = sample['image'], sample['gt']

            image = image.crop(region)
            gt = gt.crop(region)

            image = image.resize((width, height), Image.BILINEAR)
            gt = gt.resize((width, height), Image.NEAREST)

            sample['image'], sample['gt'] = image, gt
            samples[i] = sample

        return samples
