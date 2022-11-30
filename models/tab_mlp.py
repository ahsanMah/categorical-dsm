import torch
import torch.nn as nn

from models.layers import (
    GaussianFourierProjection,
    PositionalEncoding,
    TabMLPBlock,
)


class TabMLP(nn.Module):
    """Follows the style from RTDL
    The time embedding is only added once in the beginning
    """

    def __init__(self, config):
        super().__init__()

        self.ndims = ndims = config.model.ndims
        self.act = act = config.model.act
        self.embedding_size = embedding_size = config.model.time_embedding_size
        self.layers = config.model.layers
        self.dropout = dropout = config.model.dropout
        self.continuous_dim = config.data.cont_dims
        self.categorical_dims = config.data.categories
        self.num_classes = len(config.data.categories)
        self.input_dims = self.continuous_dim + sum(self.categorical_dims)

        time_encoder = GaussianFourierProjection if config.model.embedding_type == "fourier" else PositionalEncoding
        self.time_embed = nn.Sequential(
            time_encoder(embedding_size=embedding_size),
            nn.Linear(embedding_size, ndims),
            nn.SiLU(),
            nn.Linear(ndims, ndims),
        )

        self.proj = nn.Linear(self.input_dims, ndims)

        _modules = []
        for _ in range(self.layers):
            _modules.append(TabMLPBlock(ndims, ndims, act, dropout))
        # Final head to project back to input dimensions
        _modules.append(nn.Linear(ndims, self.input_dims))
        self.mlp = nn.Sequential(*_modules)

    def forward(self, x, t):
        emb = self.time_embed(t)
        x = self.proj(x) + emb
        # print(x)
        return self.mlp(x)
