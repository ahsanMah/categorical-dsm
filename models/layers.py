import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

activations = {
    "swish": F.silu,
    "mish": F.mish,
    "elu": F.elu,
    "gelu": F.gelu,
    "selu": F.selu,
    "relu": F.relu,
    "prelu": F.prelu,
    "softplus": F.softplus,
}


def get_act(name):
    if name not in activations:
        raise NotImplementedError(f"Activation function {name} does not exist!")
    return activations[name]


class PositionalEncoding(nn.Module):
    """Positional embeddings for noise levels."""

    def __init__(self, embedding_size=256, max_positions=10000):
        super().__init__()

        half_dim = embedding_size // 2
        # magic number 10000 is from transformers
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer("emb", emb)

    def forward(self, timesteps):
        x_emb = timesteps.float()[:, None] * self.emb[None, :]
        return torch.cat([torch.sin(x_emb), torch.cos(x_emb)], dim=1)


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        half_dim = embedding_size // 2
        self.W = nn.Parameter(torch.randn(half_dim) * scale, requires_grad=False)
        self.pi = torch.tensor(np.pi)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * self.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


############


class FiLMBlock(nn.Module):
    def __init__(self, time_dim_sz, num_channels, img_input=False):
        super().__init__()
        self.c = num_channels
        self.dense = nn.Linear(time_dim_sz, num_channels * 2)
        self.img_input = img_input

    def forward(self, X, time_emb):
        gamma, beta = torch.split(self.dense(time_emb), (self.c, self.c), dim=1)
        # print(gamma[:,:,None, None].shape)
        
        if self.img_input:
            return X *gamma[:, :, None, None] + beta[:, :, None, None]

        return X * gamma + beta


class ResNeXtBlock(nn.Module):
    """The ResNeXt block taken from d2l.ai."""

    def __init__(
        self,
        num_channels,
        time_emb_size,
        groups=1,
        bot_mul=1,
        use_1x1conv=True,
        strides=1,
        act="gelu",
    ):
        super().__init__()

        # !FIXME: Get this from config
        self.act = get_act(act)

        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.Conv2d(num_channels, bot_channels, kernel_size=1, stride=1)
        self.norm1 = nn.InstanceNorm2d(bot_channels)

        self.conv2 = nn.Conv2d(
            bot_channels,
            bot_channels,
            kernel_size=3,
            stride=strides,
            padding=1,
            groups=bot_channels // groups,
        )
        self.norm2 = nn.InstanceNorm2d(bot_channels)

        self.conv3 = nn.Conv2d(bot_channels, num_channels, kernel_size=1, stride=1)
        self.norm3 = nn.InstanceNorm2d(num_channels)

        if use_1x1conv:
            self.conv4 = nn.Conv2d(
                num_channels, num_channels, kernel_size=1, stride=strides
            )
            self.norm4 = nn.InstanceNorm2d(num_channels)
        else:
            self.conv4 = None

        self.film1 = FiLMBlock(time_emb_size, bot_channels)
        self.film2 = FiLMBlock(time_emb_size, bot_channels)
        self.film3 = FiLMBlock(time_emb_size, num_channels)

    def forward(self, X, time_emb):

        Y = self.act(self.norm1(self.conv1(X)))
        Y = self.film1(Y, time_emb)

        Y = self.act(self.norm2(self.conv2(Y)))
        Y = self.film2(Y, time_emb)

        Y = self.norm3(self.conv3(Y))
        Y = self.film3(Y, time_emb)

        if self.conv4:
            X = self.norm4(self.conv4(X))

        return self.act(Y + X)


class TabMLPBlock(nn.Module):
    def __init__(self, d_in, d_out, act, dropout=0.1):

        super().__init__()

        self.act = get_act(act)
        self.dense = nn.Linear(d_in, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.act(self.dense(x)))


class TabResBlockpp(nn.Module):
    def __init__(self, d_in, d_out, time_emb_sz, act="relu", dropout=0.1):

        super().__init__()

        self.norm = nn.GroupNorm(num_groups=8, num_channels=d_in)
        self.dense_1 = nn.Linear(d_in, d_out)
        self.act = get_act(act)
        self.film = FiLMBlock(time_emb_sz, d_out)
        # self.dense_cond = nn.Linear(time_emb_sz, d_out)
        self.dropout = nn.Dropout(dropout)
        self.dense_2 = nn.Linear(d_out, d_out)

    def forward(self, x, t):

        h = self.act(self.norm(x))
        h = self.dense_1(h)
        # h += self.dense_cond(t)
        h = self.film(h, t)
        h = self.dropout(h)
        h = self.dense_2(h)

        return x + h