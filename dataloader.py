import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, Omniglot, FashionMNIST
from torchvision.transforms import Compose, InterpolationMode, Lambda, Resize, ToTensor


def get_dataset(config):

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
    # FIXME: There's prolly a btter way to do this
    # Maybe have bins per class..?
    N_CATEGORIES = config.data.num_categories
    x = data.data.ravel() / 255.0
    _, BINS = np.histogram(x[::2], bins=N_CATEGORIES - 1)

    def to_1hot(x):
        x = torch.bucketize(x, torch.from_numpy(BINS))
        x = F.one_hot(x, num_classes=N_CATEGORIES)
        x = x.permute(3, 1, 2, 0).squeeze().float()
        return x

    dataset_name = config.data.dataset.lower()

    # if config.data.dataset.lower() in ["mnist", "omniglot", "fashion"]:

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
    elif dataset_name == "fashion":
        dataset = FashionMNIST
        data_transform = Compose(
            (
                ToTensor(),
                Resize((img_sz, img_sz), interpolation=InterpolationMode.BILINEAR),
                Lambda(to_1hot),
            )
        )
    else:
        raise NotImplementedError

    data = dataset(rootdir, download=True, transform=data_transform)

    num_samples = len(data)
    train_data, val_data = random_split(
        data, [int(0.9 * num_samples), int(0.1 * num_samples)]
    )

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
