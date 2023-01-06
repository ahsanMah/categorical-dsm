import logging
import pdb

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, ConcatDataset, Subset, TensorDataset, random_split
from torchvision.datasets import MNIST, FashionMNIST, Omniglot
from torchvision.transforms import Compose, InterpolationMode, Lambda, Resize, ToTensor

from configs.dataconfigs import get_config
from models.mutils import onehot_to_logit

# tabular_datasets = {
#     "adult": "adult.csv",
#     "bank": "bank.arff",
#     "chess": "chess_krkopt_zerovsall.arff",
# }
tabular_datasets = {
    "bank": "bank-additional-ful-nominal.arff",
    "chess": "chess_krkopt_zerovsall.arff",
    "census": "census.pkl",
}


def get_dataset(config, train_mode=True, seed=42, return_with_loader=True):

    generator = torch.Generator().manual_seed(seed)
    dataset_name = config.data.dataset.lower()

    if dataset_name in tabular_datasets:
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
    # Split 80,20 train, test
    # split test 50,50 into val,test
    #!FIXME: Uncomment this 
    logging.info(f"Splitting dataset with seed: {seed}")

    if dataset_name in tabular_datasets:
        inliers = data.tensors[1] == 0
        inlier_idxs = torch.argwhere(inliers).squeeze()
        outlier_idxs = torch.argwhere(~inliers).squeeze()
        logging.info(f"# Outliers: {len(outlier_idxs)}")
        inlier_ds = Subset(data, inlier_idxs)
        outlier_ds = Subset(data, outlier_idxs)
        # pdb.set_trace()
        train_ds, val_ds, test_ds = random_split(inlier_ds, [0.8, 0.1, 0.1], generator=generator)
        test_ds = ConcatDataset([test_ds, outlier_ds])
    else:
        train_ds, val_ds, test_ds = random_split(data, [0.8, 0.1, 0.1], generator=generator)

    logging.info(f"Train, Val, Test: {len(train_ds)}, {len(val_ds)}, {len(test_ds)}")

    # if train_mode and dataset_name in tabular_datasets:
    #     inlier_idxs = [idx for idx, (x, y) in enumerate(train_ds) if y == 0]
    #     train_ds = Subset(train_ds, inlier_idxs)

    if return_with_loader:
        train_ds = DataLoader(
            train_ds,
            batch_size=config.training.batch_size,
            num_workers=12,
            pin_memory=True,
        )

        val_ds = DataLoader(
            val_ds,
            batch_size=config.eval.batch_size,
            num_workers=8,
            pin_memory=True,
        )

        test_ds = DataLoader(
            test_ds,
            batch_size=config.eval.batch_size,
            num_workers=8,
            pin_memory=True,
        )

    return train_ds, val_ds, test_ds


# !TODO: Have this load raw data and labels separately
def load_dataset(name):
    str_type = lambda x: str(x, "utf-8")

    if name in ["adult"]:
        return pd.read_csv(f"data/{name}.csv").dropna()

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


# def build_tabular_ds(name):
#     raw_data = load_dataset(name)

#     to_logit = lambda x: np.log(np.clip(x, a_min=1e-5, a_max=1.0))
#     categorical_columns_selector = selector(dtype_include=object)
#     continuous_columns_selector = selector(dtype_include=[int, float])
#     categorical_features = categorical_columns_selector(raw_data)
#     continuous_features = continuous_columns_selector(raw_data)

#     encoder = OneHotEncoder(sparse=False)
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", StandardScaler(), continuous_features),
#             (
#                 "cat",
#                 make_pipeline(encoder, FunctionTransformer(to_logit)),
#                 categorical_features,
#             ),
#         ]
#     )

#     data = preprocessor.fit_transform(raw_data)
#     # Assumes last column is always label
#     category_counts = [
#         len(c)
#         for c in preprocessor.named_transformers_["cat"]
#         .named_steps["onehotencoder"]
#         .categories_
#     ]
#     label_cols = category_counts[-1]
#     X = torch.from_numpy(data[:, :-label_cols]).float()
#     y = torch.from_numpy(data[:, -label_cols:].argmax(1)).float()
#     logging.info(f"Loaded dataset: {name}, Samples: {X.shape[0]}")
#     return TensorDataset(X, y)


def build_tabular_ds(name):
    X, y, dataconfig = load_dataset(name)
    to_logit = lambda x: np.log(np.clip(x, a_min=1e-5, a_max=1.0))

    categorical_columns_selector = selector(dtype_include=object)
    continuous_columns_selector = selector(dtype_include=[int, float])
    categorical_features = categorical_columns_selector(X)
    continuous_features = continuous_columns_selector(X)

    encoder = OneHotEncoder(sparse=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), continuous_features),
            (
                "cat",
                make_pipeline(encoder, FunctionTransformer(to_logit)),
                categorical_features,
            ),
        ]
    )

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
