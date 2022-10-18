import torch.nn as nn
from models.layers import ResNeXtBlock, FiLMBlock, GaussianFourierProjection


class ResNextpp(nn.Module):
    """The ResNeXt block taken from d2l.ai."""

    def __init__(self, config, nf=16, embedding_size=16):
        super().__init__()

        self.n_channels = config.data.num_categories
        # self.nf =
        self.proj = GaussianFourierProjection(embedding_size=embedding_size)
        self.dense = nn.Linear(embedding_size * 2, embedding_size * 4)
        self.film1 = FiLMBlock(embedding_size * 4, nf)

        self.c1 = nn.Conv2d(self.n_channels, nf, kernel_size=3, stride=1)
        self.resblock1 = ResNeXtBlock(nf, embedding_size * 4, groups=2, bot_mul=2)
        self.resblock2 = ResNeXtBlock(nf, embedding_size * 4, groups=4, bot_mul=4)
        self.resblock3 = ResNeXtBlock(nf, embedding_size * 4, groups=4, bot_mul=4)
        self.resblock4 = ResNeXtBlock(nf, embedding_size * 4, groups=2, bot_mul=2)

        self.c2 = nn.ConvTranspose2d(nf, self.n_channels, kernel_size=3, stride=1)

    def forward(self, x, t):
        temb = self.dense(self.proj(t))
        x = self.film1(self.c1(x), temb)
        # print(x.shape)
        x = self.resblock1(x, temb)
        x = self.resblock2(x, temb)
        x = self.resblock3(x, temb)
        x = self.resblock4(x, temb)

        x = self.c2(x)

        return x
