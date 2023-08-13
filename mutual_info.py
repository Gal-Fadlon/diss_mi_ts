import numpy as np
import torch

def logsumexp(value, axis, keepdims=False):
    m = torch.max(value, dim=axis, keepdim=True).values
    value0 = value - m
    if not keepdims:
        m = m.squeeze(dim=axis)
    return m + torch.log(torch.sum(torch.exp(value0), dim=axis, keepdim=keepdims))

def log_density(sample, mu, logsigma):
    c = torch.tensor([np.log(2 * np.pi)], dtype=sample.dtype)

    inv_sigma = torch.exp(-logsigma)
    tmp = (sample - mu) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * logsigma + c)
