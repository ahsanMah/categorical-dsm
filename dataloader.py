import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset, Subset
from torchvision.datasets import MNIST, Omniglot, FashionMNIST
from torchvision.transforms import Compose, InterpolationMode, Lambda, Resize, ToTensor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector


def get_dataset(config, mode="train"):

    generator=torch.Generator().manual_seed(42)
    dataset_name = config.data.dataset.lower()


    if dataset_name in ["adult"]:
        data = build_tabular_ds(dataset_name)
    
    # If a torchvision dataset
    elif dataset_name in ["mnist", "omniglot", "fashion"]:
        rootdir = "/tmp/datasets"
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
        N_CATEGORIES = config.data.num_categories
        x = data.data.ravel() / 255.0
        _, BINS = np.histogram(x[::2], bins=N_CATEGORIES - 1)

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
                    Lambda(to_1hot),
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
                )
            )
        else:
            dataset = FashionMNIST
            data_transform = Compose(
                (
                    ToTensor(),
                    Resize((img_sz, img_sz), interpolation=InterpolationMode.BILINEAR),
                    Lambda(to_1hot),
                )
            )
        data = dataset(rootdir, download=True, transform=data_transform)
    else:
        raise NotImplementedError

    num_samples = len(data)
    train_data, val_data = random_split(
        data, [0.9, 0.1], generator=generator
    )

    if mode == "train":
        inlier_idxs = [idx for idx, (x,y) in enumerate(train_data) if y.argmax() == 0]
        train_data = Subset(train_data, inlier_idxs)

    train_ds = DataLoader(
        train_data,
        batch_size=config.training.batch_size,
        num_workers=8,
        pin_memory=True,
    )

    val_ds = DataLoader(
        val_data,
        batch_size=config.eval.batch_size,
        num_workers=8,
        pin_memory=True,
    )

    return train_ds, val_ds

# !TODO: Have this load raw data and labels separately
def load_dataset(name):
    if name in ["adult"]:
        return pd.read_csv(f"data/{name}.csv").dropna()

def build_tabular_ds(name):
    raw_data = load_dataset(name)

    categorical_columns_selector = selector(dtype_include=object)
    continuous_columns_selector = selector(dtype_include=[int, float])
    categorical_features = categorical_columns_selector(raw_data)
    continuous_features = continuous_columns_selector(raw_data)

    encoder = OneHotEncoder(sparse=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), continuous_features),
            ("cat", encoder, categorical_features),
            #!FIXME: put oneot to logit here
        ]
    )
    
    data = preprocessor.fit_transform(raw_data)
    # Assumes last column is always label
    category_counts = [len(c) for c in preprocessor.named_transformers_["cat"].categories_ ]
    label_cols = category_counts[-1]
    X = torch.from_numpy(data[:, :-label_cols]).float()
    y = torch.from_numpy(data[:, -label_cols:]).float()

    return TensorDataset(X,y)
