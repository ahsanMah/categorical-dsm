import torch
import torch.nn as nn

from models.layers import FiLMBlock, GaussianFourierProjection, PositionalEncoding, ResNeXtBlock


class ResNextpp(nn.Module):
    """The ResNeXt block taken from d2l.ai."""

    def __init__(self, config):
        super().__init__()

        self.nf = nf = config.model.nf
        self.act = config.model.act
        self.embedding_size = embedding_size = config.model.time_embedding_size
        self.layers = config.model.layers
        self.n_channels = config.data.num_categories
        time_encoder = (
            GaussianFourierProjection
            if config.model.embedding_type == "fourier"
            else PositionalEncoding
        )
        self.time_embed = nn.Sequential(
            time_encoder(embedding_size=embedding_size),
            nn.Linear(embedding_size, embedding_size),
            nn.SiLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.SiLU()
        )

        self.proj = nn.Conv2d(self.n_channels, nf, kernel_size=3, stride=1)

        self.blocks = []
        for channel_multiplier in self.layers:
            self.blocks.append(
                ResNeXtBlock(
                    nf,
                    embedding_size,
                    groups=channel_multiplier,
                    bot_mul=channel_multiplier,
                    act=self.act
                )
            )
        
        self.blocks = nn.ModuleList(self.blocks)

        self.final_conv = nn.ConvTranspose2d(nf, self.n_channels, kernel_size=3, stride=1)

    def forward(self, x, t):
        # t = t * (1 - 1e-5) + 1e-5
        # t = torch.log(t)
        temb = self.time_embed(t)
        x = self.proj(x)
        
        for resblock in self.blocks:
            x = resblock(x, temb)

        x = self.final_conv(x)

        return x
