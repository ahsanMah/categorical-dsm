import torch


def to_logit(probs):
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


def log_concrete_grad(x_logit, class_logits, tau):
    K = class_logits.shape[1]
    logit_noise = class_logits - tau * x_logit
    grad = -tau + tau * K * torch.softmax(logit_noise, dim=1)

    return grad


def gumbel_softmax(class_logits, tau):
    eps = 1e-20
    U = torch.rand(class_logits.shape, device=class_logits.device)
    gumbel_samples = -torch.log(-torch.log(U + eps) + eps)
    x = (gumbel_samples + class_logits) / tau
    return torch.softmax(x, dim=1).contiguous()


def gumbel_grad(x_sample, class_logits, tau, K):
    pis = class_logits.softmax(dim=1).contiguous()
    logit_sum = torch.sum(pis / torch.pow(x_sample, tau), dim=1, keepdim=True)
    denominator = logit_sum * torch.pow(x_sample, tau + 1)
    bias = (tau + 1) / x_sample

    grad = ((tau * K * pis) / denominator) - bias

    return grad
