import torch
import torch.nn.functional as F

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
    batch_sz = x_logit.shape[0]
    K = x_logit.shape[1]
    targets = log_concrete_grad(x_noisy, x_logit, tau=tau)
    loss = (scores - targets) ** 2
    loss = 0.5 * torch.sum(loss.reshape(batch_sz, -1), dim=-1)
    loss /= K

    with torch.no_grad():
        scores, targets = scores.double(), targets.double()
        rel_err = (scores - targets).abs()
        # rel_err = (rel_err / targets.abs()).mean()
        rel_err = (rel_err / torch.maximum(scores.abs(), targets.abs())).mean()
    
    return torch.mean(loss), rel_err

kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
def KL_loss(x_logit, x_noisy, model_out, tau):
    """
    x_logit: Logit probs of original sample
    x_noisy: probaility tensor of noisy image
    """
    batch_sz = x_logit.shape[0]
    K = x_logit.shape[1]
    targets = F.softmax(x_logit - tau * x_noisy, dim=1)
    loss = kl_loss_fn(F.log_softmax(model_out, dim=1), targets)

    with torch.no_grad():
        scores = -tau + tau * K * torch.softmax(model_out, dim=1)
        targets = log_concrete_grad(x_noisy, x_logit, tau=tau)
        scores, targets = scores.double(), targets.double()
        rel_err = (scores - targets).abs()
        rel_err = (rel_err / torch.maximum(scores.abs(), targets.abs())).mean()
    
    
    return loss, rel_err

def continuous_dsm_loss(noise, scores, sigmas):
    batch_sz = scores.shape[0]
    D = scores.shape[1]
    target = - noise / (sigmas ** 2)
    loss = (scores - target) ** 2
    loss = 0.5 * torch.sum(loss.reshape(batch_sz, -1), dim=-1)
    # loss = torch.mean(loss.reshape(batch_sz, -1), dim=-1)
    # loss *= sigmas[:,0] ** 2

    with torch.no_grad():
        scores, target = scores.double(), target.double()
        rel_err = (scores - target).abs()
        rel_err = (rel_err / torch.maximum(scores.abs(), target.abs())).mean()
    

    return torch.mean(loss), rel_err