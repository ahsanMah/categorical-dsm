### Adapted from lucirain's implementation
### https://github.com/lucidrains/tab-transformer-pytorch/blob/e2e8b58e7d1b453e47bc4638164e10c19fce271b/tab_transformer_pytorch/ft_transformer.py
### Including time conditioning

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat

from models.layers import FiLMBlock

# feedforward and attention


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


def FeedForward(dim, mult=4, dropout=0.0):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim),
    )


class FeedForwardpp(nn.Module):
    def __init__(self, dim, time_emb_sz, mult=4, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dense_1 = nn.Linear(dim, dim * mult * 2)
        self.act = GEGLU()
        self.cond_layer = nn.Linear(time_emb_sz, dim * mult * 2)
        self.dropout = nn.Dropout(dropout)
        self.dense_2 = nn.Linear(dim * mult, dim)

    def forward(self, x, t):
        x = self.norm(x)
        x = self.dense_1(x)
        x = self.act(x)
        # Time conditioning
        shift, scale = self.cond_layer(t[:, None, :]).chunk(2, dim=-1)
        x = torch.addcmul(shift, x, scale + 1)
        x = self.dropout(x)
        x = self.dense_2(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        q = q * self.scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        attn = sim.softmax(dim=-1)
        dropped_attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", dropped_attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out)

        return out, attn


# transformer


class Transformer(nn.Module):
    def __init__(
        self, dim, time_emb_sz, depth, heads, dim_head, attn_dropout, ff_dropout
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim, heads=heads, dim_head=dim_head, dropout=attn_dropout
                        ),
                        FeedForwardpp(dim, time_emb_sz, dropout=ff_dropout),
                    ]
                )
            )

    def forward(self, x, emb, return_attn=False):
        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = attn_out + x
            x = ff(x, emb) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)


# numerical embedder


class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, "b n -> b n 1")
        return x * self.weights + self.biases


# main class


class FTTransformer(nn.Module):
    def __init__(
        self,
        *,
        config,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head=16,
        dim_out=1,
        num_special_tokens=2,
        attn_dropout=0.0,
        ff_dropout=0.0,
    ):
        super().__init__()


        # categories related calculations
        self.categories = categories = config.data.categories
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings
        # each categorical softmax-vector will get projected into an embedding space
        self.categorical_embeds = torch.nn.ModuleDict(
            {
                "proj_in": torch.nn.ModuleList(
                    [torch.nn.Linear(c, dim) for c in categories]
                ),
                "proj_out": torch.nn.ModuleList(
                    [torch.nn.Linear(dim, c) for c in categories]
                ),
            }
        )

        # continuous

        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        # transformer

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )

        # to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim), nn.ReLU(), nn.Linear(dim, dim_out)
        )

    def forward(self, x, t, return_attn=False):
        # assert (
        #     x_categ.shape[-1] == self.num_categories
        # ), f"you must pass in {self.num_categories} values for your categories input"

        x_numer, x_categ = torch.split(
            x, [self.num_continuous, self.num_unique_categories]
        )

        xs = []

        # add numerically embedded tokens
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)
            xs.append(x_numer)

        # categorical embedded tokens
        x_cats = torch.split(x_categ, self.categories, dim=1)
        x_cat_embs = torch.stack(
            [embedder(cat) for cat, embedder in zip(x_cats, self.categorical_embeds)], dim=1
        )

        # concat categorical and numerical

        x = torch.cat(xs, dim=1)

        # append cls tokens
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # attend

        x, attns = self.transformer(x, return_attn=True)

        # get cls token

        x = x[:, 0]

        # out in the paper is linear(relu(ln(cls)))

        logits = self.to_logits(x)

        if not return_attn:
            return logits

        return logits, attns
