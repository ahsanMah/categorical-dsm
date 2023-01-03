import torch
import torch.nn as nn

from models.layers import (
    GaussianFourierProjection,
    PositionalEncoding,
    TabResBlockpp,
)


class TabResNet(nn.Module):
    """
    The time embedding is added after each block
    """

    def __init__(self, config):
        super().__init__()

        self.ndims = ndims = config.model.ndims
        self.act = act = config.model.act
        self.embedding_size = embedding_size = config.model.time_embedding_size
        self.layers = config.model.layers
        self.dropout = dropout = config.model.dropout
        self.continuous_dim = config.data.numerical_features
        self.categorical_dims = config.data.categories
        self.num_classes = len(config.data.categories)
        self.input_dims = self.continuous_dim + sum(self.categorical_dims)

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

        self.proj = nn.Linear(self.input_dims, ndims)

        _modules = []
        for _ in range(self.layers):
            _modules.append(
                TabResBlockpp(ndims, ndims, embedding_size, act=act, dropout=dropout)
            )
        self.resnet = nn.ModuleList(_modules)

        # Final head to project back to input dimensions
        self.final_head = nn.Sequential(
            # nn.GroupNorm(num_groups=8, num_channels=ndims),
            # nn.ReLU(),
            nn.LayerNorm(ndims),
            nn.LeakyReLU(),
            nn.Linear(ndims, self.input_dims),
        )

    def forward(self, x, t):
        emb = self.time_embed(t)
        x = self.proj(x)

        for m in self.resnet:
            x = m(x, emb)

        return self.final_head(x)
