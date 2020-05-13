import torch
import torch.nn.functional as F
from tqdm import tqdm

# --------------------
# Loss functions
# --------------------

def discretized_mix_logistic_loss(l, x, n_bits):
    """ log likelihood for mixture of discretized logistics
    Args
        l -- model output tensor of shape (B, 10*n_mix, H, W), where for each n_mix there are
                3 params for means, 3 params for coefficients, 3 params for logscales, 1 param for logits
        x -- data tensor of shape (B, C, H, W) with values in model space [-1, 1]
    """
    # shapes
    B, C, H, W = x.shape
    n_mix = l.shape[1] // (1 + 3*C)

    # unpack params of mixture of logistics
    logits = l[:, :n_mix, :, :]                         # (B, n_mix, H, W)
    l = l[:, n_mix:, :, :].reshape(B, 3*n_mix, C, H, W)
    means, logscales, coeffs = l.split(n_mix, 1)        # (B, n_mix, C, H, W)
    logscales = logscales.clamp(min=-7)
    coeffs = coeffs.tanh()

    # adjust means of channels based on preceding subpixel (cf PixelCNN++ eq 3)
    x  = x.unsqueeze(1).expand_as(means)
    if C!=1:
        m1 = means[:, :, 0, :, :]
        m2 = means[:, :, 1, :, :] + coeffs[:, :, 0, :, :] * x[:, :, 0, :, :]
        m3 = means[:, :, 2, :, :] + coeffs[:, :, 1, :, :] * x[:, :, 0, :, :] + coeffs[:, :, 2, :, :] * x[:, :, 1, :, :]
        means = torch.stack([m1, m2, m3], 2)  # out (B, n_mix, C, H, W)

    # log prob components
    scales = torch.exp(-logscales)
    plus = scales * (x - means + 1/(2**n_bits-1))
    minus = scales * (x - means - 1/(2**n_bits-1))

    # partition the logistic pdf and cdf for x in [<-0.999, mid, >0.999]
    # 1. x<-0.999 ie edge case of 0 before scaling
    cdf_minus = torch.sigmoid(minus)
    log_one_minus_cdf_minus = - F.softplus(minus)
    # 2. x>0.999 ie edge case of 255 before scaling
    cdf_plus = torch.sigmoid(plus)
    log_cdf_plus = plus - F.softplus(plus)
    # 3. x in [-.999, .999] is log(cdf_plus - cdf_minus)

    # compute log probs:
    # 1. for x < -0.999, return log_cdf_plus
    # 2. for x > 0.999,  return log_one_minus_cdf_minus
    # 3. x otherwise,    return cdf_plus - cdf_minus
    log_probs = torch.where(x < -0.999, log_cdf_plus,
                            torch.where(x > 0.999, log_one_minus_cdf_minus,
                                        torch.log((cdf_plus - cdf_minus).clamp(min=1e-12))))
    log_probs = log_probs.sum(2) + F.log_softmax(logits, 1) # log_probs sum over channels (cf eq 3), softmax over n_mix components (cf eq 1)

    # marginalize over n_mix components and return negative log likelihood per data point
    return - log_probs.logsumexp(1).sum([1,2])  # out (B,)

loss_fn = discretized_mix_logistic_loss

# --------------------
# Sampling and generation functions
# --------------------

def sample_from_discretized_mix_logistic(l, image_dims):
    # shapes
    B, _, H, W = l.shape
    C = image_dims[0]#3
    n_mix = l.shape[1] // (1 + 3*C)

    # unpack params of mixture of logistics
    logits = l[:, :n_mix, :, :]
    l = l[:, n_mix:, :, :].reshape(B, 3*n_mix, C, H, W)
    means, logscales, coeffs = l.split(n_mix, 1)  # each out (B, n_mix, C, H, W)
    logscales = logscales.clamp(min=-7)
    coeffs = coeffs.tanh()

    # sample mixture indicator
    argmax = torch.argmax(logits - torch.log(-torch.log(torch.rand_like(logits).uniform_(1e-5, 1 - 1e-5))), dim=1)
    sel = torch.eye(n_mix, device=logits.device)[argmax]
    sel = sel.permute(0,3,1,2).unsqueeze(2)  # (B, n_mix, 1, H, W)

    # select mixture components
    means = means.mul(sel).sum(1)
    logscales = logscales.mul(sel).sum(1)
    coeffs = coeffs.mul(sel).sum(1)

    # sample from logistic using inverse transform sampling
    u = torch.rand_like(means).uniform_(1e-5, 1 - 1e-5)
    x = means + logscales.exp() * (torch.log(u) - torch.log1p(-u))  # logits = inverse logistic

    if C==1:
        return x.clamp(-1,1)
    else:
        x0 = torch.clamp(x[:,0,:,:], -1, 1)
        x1 = torch.clamp(x[:,1,:,:] + coeffs[:,0,:,:] * x0, -1, 1)
        x2 = torch.clamp(x[:,2,:,:] + coeffs[:,1,:,:] * x0 + coeffs[:,2,:,:] * x1, -1, 1)
        return torch.stack([x0, x1, x2], 1)  # out (B, C, H, W)

'''
def generate_fn(model, n_samples, image_dims, device, h=None):
    out = torch.zeros(n_samples, *image_dims, device=device)
    with tqdm(total=(image_dims[1]*image_dims[2]), desc='Generating {} images'.format(n_samples)) as pbar:
        for yi in range(image_dims[1]):
            for xi in range(image_dims[2]):
                l = model(out, h)
                out[:,:,yi,xi] = sample_from_discretized_mix_logistic(l, image_dims)[:,:,yi,xi]
                pbar.update()
    return out
'''

def generate_fn(model, data_loader, n_samples, image_dims, device, h=None):
    out, info = next(iter(data_loader))
    print("The generate size is:", out.size())
    out[:, 1, :, :] = torch.zeros(128,128)
    out = out.to(device)
    with tqdm(total=(image_dims[1]*image_dims[2]), desc='Generating {} images'.format(out.size(0))) as pbar:
        for yi in range(image_dims[1]):
            for xi in range(image_dims[2]):
                logits = model(out, h)
                sample = sample_from_discretized_mix_logistic(logits, image_dims)[:,:,yi,xi]
                out[:,1,yi,xi] = sample[:,1]
                pbar.update()
                    
    return out, info[0]