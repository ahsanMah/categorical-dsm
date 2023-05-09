import random
from typing import Tuple

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img: torch.Tensor, size: int):
    shape = img.shape[-2:]
    min_size = min(shape)
    if min_size < size:
        oh, ow = shape
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=0)
    return img


class RandomResize(torch.nn.Module):
    def __init__(self, min_size, max_size=None):
        super().__init__()
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def forward(
        self, image_and_target: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image, target = image_and_target
        size = int(torch.randint(self.min_size, self.max_size + 1, size=(1,)).item()),
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, flip_prob):
        super().__init__()
        self.flip_prob = flip_prob

    def forward(
        self, image_and_target: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image, target = image_and_target
        if torch.rand(1) < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop(T.RandomCrop):
    def __init__(self, size):
        super().__init__(size)
        # print("RandCrop Size:", self.size)
        self.size = size

    def forward(
        self, image_and_target: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image, target = image_and_target
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size)
        crop_params = self.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(
        self, image_and_target: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image, target = image_and_target
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
