import logging

# import pdb
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Subset,
    TensorDataset,
    random_split,
)
from torchvision.datasets import MNIST, FashionMNIST, Omniglot, VOCSegmentation
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)
from torchvision.transforms import Compose, InterpolationMode, Lambda, Resize, ToTensor
from tqdm.auto import tqdm

import models.segmentation.module_transforms as SegT
from configs.dataconfigs import get_config
from models.mutils import onehot_to_logit
from models.segmentation.presets import SegmentationEval, SegmentationTrain

tabular_datasets = {
    "bank": "bank-additional-ful-nominal.arff",
    "chess": "chess_krkopt_zerovsall.arff",
    "census": "census.pkl",
    "probe": "kddcup99-corrected-probevsnormal-nominal-cleaned.arff",
    "u2r": "kddcup99-corrected-u2rvsnormal-nominal-cleaned.arff",
    "solar": "solar-flare_FvsAll-cleaned.arff",
    "cmc": "cmc-nominal.arff",
}


def get_dataset(config, train_mode=True, return_with_loader=True, return_logits=True):

    generator = torch.Generator().manual_seed(config.seed)
    dataset_name = config.data.dataset.lower()
    rootdir = "/tmp/datasets"

    if dataset_name in tabular_datasets:
        data = build_tabular_ds(dataset_name, return_logits=return_logits)

    # If a torchvision dataset
    elif dataset_name in ["voc"]:
        img_sz = config.data.image_size

        if config.data.cached:
            # print()
            if train_mode:
                preprocessing = TrainTransform(
                    out_size=img_sz,
                    base_size=520,
                    crop_size=480,
                    to_logits=config.data.logits,
                )

            else:
                preprocessing = SegmentationEval(
                    out_size=img_sz, to_logits=config.data.logits
                )
            data = CachedVOCSegmentation(
                root=rootdir,
                download=False,
                image_set="train",  # if train_mode else "val",
                transforms=preprocessing,
            )
        else:
            if train_mode:
                preprocessing = SegmentationTrain(
                    out_size=img_sz,
                    base_size=520,
                    crop_size=480,
                    to_logits=config.data.logits,
                )

            else:
                preprocessing = SegmentationEval(
                    out_size=img_sz, to_logits=config.data.logits
                )
            data = VOCSegmentation(
                root=rootdir,
                download=False,
                image_set=config.data.image_set,
                transforms=preprocessing,
            )

    elif dataset_name in ["mnist", "omniglot", "fashion"]:
        img_sz = config.data.image_size

        data = MNIST(
            rootdir,
            download=True,
            transform=Compose(
                (
                    ToTensor(),
                    Resize((img_sz, img_sz), interpolation=InterpolationMode.BILINEAR),
                )
            ),
        )
        # FIXME: There's prolly a better way to do this
        # Maybe have bins per class..?
        N_CATEGORIES = config.data.categorical_channels

        if os.path.exists(f"data/mnist_bins={N_CATEGORIES}.npz"):
            BINS = np.load(f"data/mnist_bins={N_CATEGORIES}.npz")["arr_0"]
        else:
            x = data.data.ravel() / 255.0
            _, BINS = np.histogram(x, bins=N_CATEGORIES - 1)
            np.savez_compressed(f"data/mnist_bins={N_CATEGORIES}.npz", BINS)

        def to_1hot(x):
            x = torch.bucketize(x, torch.from_numpy(BINS))
            x = F.one_hot(x, num_classes=N_CATEGORIES)
            x = x.permute(3, 1, 2, 0).squeeze().float()
            return x

        if dataset_name == "mnist":
            dataset = MNIST
            data_transform = Compose(
                (
                    ToTensor(),
                    Resize((img_sz, img_sz), interpolation=InterpolationMode.BILINEAR),
                    Lambda(to_1hot),
                    Lambda(onehot_to_logit),
                )
            )
        elif dataset_name == "omniglot":
            dataset = Omniglot
            data_transform = Compose(
                (
                    ToTensor(),
                    lambda x: 1 - x,
                    Resize((img_sz, img_sz), interpolation=InterpolationMode.BILINEAR),
                    Lambda(to_1hot),
                    Lambda(onehot_to_logit),
                )
            )
        else:
            dataset = FashionMNIST
            data_transform = Compose(
                (
                    ToTensor(),
                    Resize((img_sz, img_sz), interpolation=InterpolationMode.BILINEAR),
                    Lambda(to_1hot),
                    Lambda(onehot_to_logit),
                )
            )
        data = dataset(
            rootdir, train=train_mode, download=True, transform=data_transform
        )
    else:
        raise NotImplementedError

    # Subset inlier only
    logging.info(f"Splitting dataset with seed: {config.seed}")

    # Subset inlier only
    # Split 80,10,10 train, val, test
    # Combine test and outlier
    if dataset_name in tabular_datasets:
        inliers = data.tensors[1] == 0
        inlier_idxs = torch.argwhere(inliers).squeeze()
        outlier_idxs = torch.argwhere(~inliers).squeeze()
        logging.info(f"# Outliers: {len(outlier_idxs)}")
        inlier_ds = Subset(data, inlier_idxs)
        outlier_ds = Subset(data, outlier_idxs)
        # pdb.set_trace()
        train_ds, val_ds, test_ds = random_split(
            inlier_ds, [0.8, 0.1, 0.1], generator=generator
        )
        test_ds = ConcatDataset([test_ds, outlier_ds])
    else:
        train_ds, val_ds = random_split(data, [0.9, 0.1], generator=generator)
        test_ds = val_ds  # WONT BE USED

    logging.info(f"Train, Val, Test: {len(train_ds)}, {len(val_ds)}, {len(test_ds)}")

    # if train_mode and dataset_name in tabular_datasets:
    #     inlier_idxs = [idx for idx, (x, y) in enumerate(train_ds) if y == 0]
    #     train_ds = Subset(train_ds, inlier_idxs)

    if return_with_loader:
        train_ds = DataLoader(
            train_ds,
            batch_size=config.training.batch_size,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=8,
            shuffle=train_mode,
        )

        val_ds = DataLoader(
            val_ds,
            batch_size=config.eval.batch_size,
            num_workers=6,
            pin_memory=True,
        )

        test_ds = DataLoader(
            test_ds,
            batch_size=config.eval.batch_size,
            num_workers=8,
            pin_memory=True,
        )

    return train_ds, val_ds, test_ds


def load_dataset(name):
    str_type = lambda x: str(x, "utf-8")

    # AD_nominal
    # dtype = all categorical
    # Anomaly: AD

    # AID
    # dtype = all categorical
    # Anomaly: active
    basedir = "data/categorical_data_outlier_detection/"
    dataconfig = get_config(name)
    label = dataconfig.label_column

    if name == "census":
        df = pd.read_pickle(basedir + tabular_datasets[name])
    elif name == "nursery":
        df = pd.read_csv(f"data/nursery.csv")
        labels = df[label]
        drop_mask = np.logical_or(labels == "not_recom", labels == "very_recom")
        labels = labels[drop_mask]
        df = df[drop_mask]
    else:
        data, metadata = arff.loadarff(basedir + tabular_datasets[name])
        df = pd.DataFrame(data).applymap(str_type)

    X = df.drop(
        columns=label,
    )
    y = np.zeros(len(df[label]), dtype=np.float32)
    ano_idxs = df[label] == dataconfig.anomaly_label
    y[ano_idxs] = 1.0
    # print(y)
    return X, y.squeeze(), dataconfig


def build_tabular_ds(name, return_logits=True):
    X, y, dataconfig = load_dataset(name)
    to_logit = lambda x: np.log(np.clip(x, a_min=1e-5, a_max=1.0))

    categorical_columns_selector = selector(dtype_include=object)
    continuous_columns_selector = selector(dtype_include=[int, float])
    categorical_features = categorical_columns_selector(X)
    continuous_features = continuous_columns_selector(X)

    cat_processor = [OneHotEncoder(sparse=False)]
    if return_logits:
        cat_processor.append(FunctionTransformer(to_logit))
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), continuous_features),
            (
                "cat",
                make_pipeline(*cat_processor),
                categorical_features,
            ),
        ]
    )

    if name in ["probe", 'nursery']:
        # Some categories only appear in outliers ...
        # so preprocessor needs to knwo them
        preprocessor.fit(X)
    else:
        # Only fit on inliers
        preprocessor.fit(X[y == 0])

    categories = [
        len(x)
        for x in preprocessor.named_transformers_["cat"]
        .named_steps["onehotencoder"]
        .categories_
    ]
    assert categories == dataconfig.categories
    assert len(continuous_features) == dataconfig.numerical_features

    X = preprocessor.transform(X)

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    logging.info(f"Loaded dataset: {name}, Shape: {X.shape}")

    return TensorDataset(X, y)


class CachedVOCSegmentation(torch.utils.data.Dataset):
    def __init__(self, root, image_set="train", download=False, transforms=None):
        self.rootdir = root
        self.image_set = image_set
        self.transforms = transforms

        self.cache = []
        self.voc = VOCSegmentation(
            root=root,
            download=download,
            image_set=image_set,
            transforms=None,
        )

        logging.info(f"Loading images from {image_set} set")
        for idx in tqdm(range(len(self.voc))):
            img, target = self.voc[idx]
            # img = Image.open(self.voc.images[idx]).convert("RGB")
            # target = Image.open(self.voc.masks[idx])
            # print(img.size, target.size)

            img = T.functional.pil_to_tensor(img)
            img = T.functional.convert_image_dtype(img, dtype=torch.float32)
            target = torch.as_tensor(np.array(target)[None, ...], dtype=torch.int64)
            # print(img.shape, target.shape)
            # break

            self.cache.append((img, target))

        logging.info(f"Loaded {len(self.cache)} images")

    def __getitem__(self, idx):
        return self.transforms(*self.cache[idx])

    def __len__(self):
        return len(self.cache)


class MultiSequential(torch.nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        for module in self:
            x = module(x)
        return x


class TrainTransform(torch.nn.Module):
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
        super().__init__()

        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)
        self.out_sz = out_size

        trans = [SegT.RandomResize(min_size, max_size)]
        # trans = []
        if hflip_prob > 0:
            trans.append(SegT.RandomHorizontalFlip(hflip_prob))
        trans.extend(
            [
                SegT.RandomCrop(crop_size),
                SegT.RandomResize(self.out_sz, self.out_sz),
                SegT.Normalize(mean=mean, std=std),
            ]
        )
        logging.info("Compiling transforms...")
        self.transforms = MultiSequential(*trans)
        self.transforms = torch.jit.script(self.transforms)
        logging.info("Completed.")
        # self.to_onehot = partial(F.one_hot, num_classes=21)
        def build_one_hot_transform(to_logits=to_logits):
            if to_logits:

                @torch.jit.script
                def to_onehot(target):
                    target[target == 255] = 0
                    target = F.one_hot(target, num_classes=21).squeeze().float()
                    target = target.permute(2, 0, 1)
                    target = torch.log(torch.clamp(target, min=1e-5, max=1.0))
                    return target

            else:

                @torch.jit.script
                def to_onehot(target):
                    target[target == 255] = 0
                    target = F.one_hot(target, num_classes=21).squeeze().float()
                    target = target.permute(2, 0, 1)
                    return target

            return to_onehot

        self.to_onehot = build_one_hot_transform()

    def __call__(self, img, target):
        img, target = self.transforms((img, target))
        target = self.to_onehot(target)
        img = torch.cat((img, target), dim=0)
        return img, 0


