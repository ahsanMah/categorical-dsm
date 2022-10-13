import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, InterpolationMode, Lambda, Resize, ToTensor


def get_dataset(config):

    if config.data.dataset.lower() == "mnist":

        data = MNIST(
            "/tmp/mnist",
            download=True,
            transform=Compose(
                (ToTensor(), Resize((8, 8), interpolation=InterpolationMode.BILINEAR))
            ),
        )
        # FIXME: There's prolly a btter way to do this
        # Maybe have bins per class..?
        N_CATEGORIES = config.data.num_categories
        x = data.data.ravel() / 255.0
        _, BINS = np.histogram(x[::4], bins=N_CATEGORIES - 1)

        def to_1hot(x):
            x = torch.bucketize(x, torch.from_numpy(BINS))
            x = F.one_hot(x, num_classes=N_CATEGORIES)
            x = x.permute(3, 1, 2, 0).squeeze().float()
            return x

        data = MNIST(
            "/tmp/mnist",
            download=True,
            transform=Compose(
                (
                    ToTensor(),
                    Resize((8, 8), interpolation=InterpolationMode.BILINEAR),
                    Lambda(to_1hot),
                )
            ),
        )
        num_samples = data.data.shape[0]
        train_data, val_data = random_split(
            data, [int(0.9 * num_samples), int(0.1 * num_samples)]
        )

        train_ds = torch.utils.data.DataLoader(
            train_data, batch_size=config.training.batch_size
        )

        val_ds = torch.utils.data.DataLoader(
            val_data, batch_size=config.eval.batch_size
        )

    else:
        raise NotImplementedError

    return train_ds, val_ds
