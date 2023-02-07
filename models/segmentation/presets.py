import torch
from . import transforms as T
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from functools import partial


class SegmentationPresetTrain:
    def __init__(
        self,
        *,
        base_size,
        crop_size,
        hflip_prob=0.5,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend(
            [
                T.RandomCrop(crop_size),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(
        self, *, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    ):
        self.transforms = T.Compose(
            [
                T.RandomResize(base_size, base_size),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationTrain:
    def __init__(
        self,
        *,
        out_size,
        base_size,
        crop_size,
        hflip_prob=0.5,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        to_logits=False,
    ):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)
        self.out_sz = out_size

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend(
            [
                T.RandomCrop(crop_size),
                T.RandomResize(self.out_sz, self.out_sz),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )
        self.transforms = T.Compose(trans)
        # self.to_onehot = partial(F.one_hot, num_classes=21)
        def build_one_hot_transform(to_logits=to_logits):
            # print('TO LOGITS:', to_logits)
            # @torch.jit.script
            def to_onehot(target):
                target = F.one_hot(target,  num_classes=21).squeeze().float()
                target = target.permute(2, 0, 1)
                if to_logits:
                    target = torch.log(torch.clamp(target, min=1e-5, max=1.0))
                
                return target
            return to_onehot
        
        self.to_onehot = build_one_hot_transform()

    def __call__(self, img, target):
        img, target = self.transforms(img, target)
        target[target == 255] = 0
        target = self.to_onehot(target)
        # print(img.shape, target.shape)
        img = torch.cat((img, target), dim=0)
        return img, 0

class SegmentationEval:
    def __init__(
        self,
        *,
        out_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        to_logits=False,
    ):

        self.out_sz = out_size
        
        self.transforms = T.Compose(
            [
                T.Resize(self.out_sz, self.out_sz),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )
        # self.to_onehot = partial(F.one_hot, num_classes=21)
        
        def build_one_hot_transform(to_logits=to_logits):
            def to_onehot(target):
                target = F.one_hot(target,  num_classes=21).squeeze().float()
                target = target.permute(2, 0, 1)
                if to_logits:
                    target = torch.log(torch.clamp(target, min=1e-5, max=1.0))
                
                return target
            return to_onehot
        
        self.to_onehot = build_one_hot_transform()


    def __call__(self, img, target):
        img, target = self.transforms(img, target)
        target[target == 255] = 0
        target = self.to_onehot(target)
        img = torch.cat((img, target), dim=0)
        
        return img, 0