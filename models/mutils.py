import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.tab_resnet import TabResNet
from models.tab_mlp import TabMLP
from models.resnext import ResNextpp
from models.ncsnpp import NCSNpp
from models.ft_transformer_pp import FTTransformer

optimizers = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "RAdam": torch.optim.RAdam,
}
models = {
    "tab-transformer": FTTransformer,
    "tab-resnet": TabResNet,
    "tab-mlp": TabMLP,
    "resnext": ResNextpp,
    "ncsn++": NCSNpp,
}


"""
Taken from Score SDE codebase
https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/models/layers.py#L54
"""


def variance_scaling(
    scale, mode, distribution, in_axis=1, out_axis=0, dtype=torch.float32, device="cpu"
):
    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
                "invalid mode for variance scaling initializer: {}".format(mode)
            )
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (
                torch.rand(*shape, dtype=dtype, device=device) * 2.0 - 1.0
            ) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init


def build_default_init_fn(scale=1.0):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    init_fn = variance_scaling(scale, "fan_avg", "uniform")

    def init_weights(module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            with torch.no_grad():
                module.weight.data = init_fn(module.weight.shape)

            if module.bias is not None:
                module.bias.data.zero_()

            # print(f"Initialized {module.__class__.__name__} with {init_fn.__name__}.")

    return init_weights


def get_model(config):
    name = config.model.name
    if name not in models:
        raise NotImplementedError(f"Model {name} does not exist!")

    return models[name](config)


def get_optimizer(name):
    if name not in optimizers:
        raise NotImplementedError(f"Optimizer {name} does not exist!")

    return optimizers[name]


def onehot_to_logit(x_hot, eps=1e-5):
    return torch.log(torch.clamp(x_hot, min=eps, max=1.0))


def log_concrete_sample(class_logits: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Log version of concrete/gumbel sampling routine taken from
    [LINK]
    Working in the log domain helps improve stability.

    Args:
        class_logits: Unnormalized class "probabilites" (logits from 1-hot vectors)
        tau: Smoothing factor

    Returns:
        torch.Tensor: Continuous samples
    """
    eps = 1e-20

    # # For debugging
    # g=torch.Generator(device=class_logits.device)
    # g.manual_seed(43)
    # U = torch.rand(class_logits.shape, generator=g, device=class_logits.device)

    U = torch.rand_like(class_logits)
    gumbel_samples = -torch.log(-torch.log(U + eps) + eps)
    x = (gumbel_samples + class_logits) / tau
    x = x - torch.logsumexp(x, dim=1, keepdim=True)
    return x


def gumbel_softmax(class_logits: torch.Tensor, tau: torch.Tensor):
    """Generates smoothed samples from probability logits.
    In practice `class_logits` will be one-hot tensors.
    This function thus spits out a smoothed representation

    Args:
        class_logits: Unnormalized class probabilites
        tau: Smoothing factor

    Returns:
        Continuous samples
    """
    eps = 1e-20
    U = torch.rand(class_logits.shape, device=class_logits.device)
    gumbel_samples = -torch.log(-torch.log(U + eps) + eps)
    x = (gumbel_samples + class_logits) / tau
    return torch.softmax(x, dim=1).contiguous()


def get_taus(config) -> torch.Tensor:
    """Generate taus with an exponential schedule seen in NCSN

    Args:
        config: Configuration file that lets us know min/max

    Returns:
        taus: Tensor of taus in *descending* order
    """
    taus = np.exp(
        np.linspace(
            np.log(config.model.tau_max),
            np.log(config.model.tau_min),
            config.model.num_scales,
            dtype=np.float32,
        )
    )
    return taus


def get_sigmas(config) -> torch.Tensor:
    """Generate sigmas with an exponential schedule seen in NCSN

    Args:
        config: Configuration file that lets us know min/max

    Returns:
        sigmas: Tensor of sigmas in *descending* order
    """
    sigmas = np.exp(
        np.linspace(
            np.log(config.model.sigma_max),
            np.log(config.model.sigma_min),
            config.model.num_scales,
            dtype=np.float32,
        )
    )
    return sigmas
