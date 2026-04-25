import random

import torch
from torchvision import transforms


class Compose:
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list

    def __call__(self, image, target):
        for t in self.transforms_list:
            image, target = t(image, target)
        return image, target


class ColorJitterOnly:
    def __init__(self, brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05):
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, image, target):
        return self.color_jitter(image), target


class RandomHorizontalFlipWithBoxes:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target

        _, _, w = image.shape
        image = torch.flip(image, dims=[2])

        boxes = target["boxes"]
        if boxes.numel() == 0:
            return image, target

        xmin, ymin, xmax, ymax = boxes.unbind(dim=1)
        new_xmin = w - xmax
        new_xmax = w - xmin
        boxes = torch.stack([new_xmin, ymin, new_xmax, ymax], dim=1)
        target["boxes"] = boxes
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        return image, target


def get_conservative_train_transforms():
    return Compose(
        [
            ColorJitterOnly(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
            RandomHorizontalFlipWithBoxes(p=0.5),
        ]
    )
