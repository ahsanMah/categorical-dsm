import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
        self.pi = torch.tensor(np.pi)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * self.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FiLMBlock(nn.Module):
    def __init__(self, time_dim_sz, num_channels):
        super().__init__()
        self.c = num_channels
        self.dense = nn.Linear(time_dim_sz, num_channels * 2)

    def forward(self, X, time_emb):
        gamma, beta = torch.split(self.dense(time_emb), (self.c, self.c), dim=1)
        # print(gamma[:,:,None, None].shape)
        return X * gamma[:, :, None, None] + beta[:, :, None, None]


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
    ):
        super().__init__()

        # !FIXME: Get this from config
        self.act = F.gelu

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
