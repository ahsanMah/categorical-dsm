import torch
import numpy as np


def prob_to_logit(probs):
    eps = 1e-5
    l = torch.clamp(torch.log(probs), min=np.log(eps))
    return l


def log_concrete_sample(class_logits: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Log version of concrete/gumbel sampling routine taken from
    [LINK]
    Working in the log domain helps improve stability.

    Args:
        class_logits: Unnormalized class probabilites
        tau: Smoothing factor

    Returns:
        torch.Tensor: Continuous samples
    """

    eps = 1e-20
    U = torch.rand(class_logits.shape, device=class_logits.device)
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
        )
    )
    return taus
