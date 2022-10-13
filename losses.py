import torch


def log_concrete_grad(x_logit, class_logits, tau):
    K = class_logits.shape[1]
    logit_noise = class_logits - tau * x_logit
    grad = -tau + tau * K * torch.softmax(logit_noise, dim=1)

    return grad


def gumbel_grad(x_sample, class_logits, tau, K):
    pis = class_logits.softmax(dim=1).contiguous()
    logit_sum = torch.sum(pis / torch.pow(x_sample, tau), dim=1, keepdim=True)
    denominator = logit_sum * torch.pow(x_sample, tau + 1)
    bias = (tau + 1) / x_sample

    grad = ((tau * K * pis) / denominator) - bias

    return grad


def categorical_dsm_loss(x_logit, x_noisy, scores, tau):
    """
    x_logit: Logit probs of original sample
    x_noisy: probaility tensor of noisy image
    """
    targets = log_concrete_grad(x_noisy, x_logit, tau=tau)
    loss = 0.5 * (scores - targets) ** 2

    return torch.mean(loss)
