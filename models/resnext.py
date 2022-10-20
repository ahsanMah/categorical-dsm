import torch.nn as nn
from models.layers import ResNeXtBlock, FiLMBlock, GaussianFourierProjection


class ResNextpp(nn.Module):
    """The ResNeXt block taken from d2l.ai."""

    def __init__(self, config):
        super().__init__()

        self.nf = nf = config.model.nf
        self.embedding_size = embedding_size = config.model.time_embedding_size
        self.layers = config.model.layers
        self.n_channels = config.data.num_categories
        # self.nf =
        self.proj = GaussianFourierProjection(embedding_size=embedding_size)
        self.dense = nn.Linear(embedding_size * 2, embedding_size * 4)
        self.film1 = FiLMBlock(embedding_size * 4, nf)

        self.c1 = nn.Conv2d(self.n_channels, nf, kernel_size=3, stride=1)

        self.blocks = []
        for channel_multiplier in self.layers:
            self.blocks.append(
                ResNeXtBlock(
                    nf,
                    embedding_size * 4,
                    groups=channel_multiplier,
                    bot_mul=channel_multiplier,
                )
            )
        
        self.blocks = nn.ModuleList(self.blocks)

        self.c2 = nn.ConvTranspose2d(nf, self.n_channels, kernel_size=3, stride=1)

    def forward(self, x, t):
        temb = self.dense(self.proj(t))
        x = self.film1(self.c1(x), temb)
        
        for resblock in self.blocks:
            x = resblock(x, temb)

        x = self.c2(x)

        return x
