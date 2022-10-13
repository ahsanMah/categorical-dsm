import torch
import numpy as np


def prob_to_logit(probs):
    eps = 1e-5
    l = torch.clamp(torch.log(probs), min=np.log(eps))
    return l


def log_concrete_sample(class_logits, tau):
    eps = 1e-20
    U = torch.rand(class_logits.shape, device=class_logits.device)
    gumbel_samples = -torch.log(-torch.log(U + eps) + eps)

    x = (gumbel_samples + class_logits) / tau
    x = x - torch.logsumexp(x, dim=1, keepdim=True)
    return x


def gumbel_softmax(class_logits, tau):
    eps = 1e-20
    U = torch.rand(class_logits.shape, device=class_logits.device)
    gumbel_samples = -torch.log(-torch.log(U + eps) + eps)
    x = (gumbel_samples + class_logits) / tau
    return torch.softmax(x, dim=1).contiguous()
