import torch
from math import exp

import torch.nn as nn
import torch.nn.functional as F

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs

    return ret

def grad_x(img):
    (_, _, _, width) = img.size()
    
    img2 = torch.narrow(img, 3, 1, width-1)
    img1 = torch.narrow(img, 3, 0, width-1)
    
    grad_img = img2 - img1
    return grad_img

def grad_y(img):
    (_, _, height, _) = img.size()
    
    img2 = torch.narrow(img, 2, 1, height-1)
    img1 = torch.narrow(img, 2, 0, height-1)
    
    grad_img = img2 - img1
    return grad_img


class MaskedL1(nn.Module):
    # https://discuss.pytorch.org/t/masked-loss-on-a-batch-of-images/2214
    def __init__(self):
        super(MaskedL1, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, varA, varB, mask):
        loss = self.criterion(varA*mask, varB*mask)
        return loss


class MaskedL1Grad(nn.Module):
    # https://discuss.pytorch.org/t/masked-loss-on-a-batch-of-images/2214
    def __init__(self):
        super(MaskedL1Grad, self).__init__()
        self.criterion = MaskedL1()

    def forward(self, varA, varB, mask):
        # make sure mask is exactly the same size as gradient images
        (_, _, height, width) = mask.size()
        m_x2 = torch.narrow(mask, 3, 1, width - 1)
        m_x1 = torch.narrow(mask, 3, 0, width - 1)
        m_y2 = torch.narrow(mask, 2, 1, height - 1)
        m_y1 = torch.narrow(mask, 2, 0, height - 1)
        mask_x = m_x2 * m_x1
        mask_y = m_y2 * m_y1

        loss = self.criterion(grad_x(varA), grad_x(varB), mask_x) \
            + self.criterion(grad_y(varA), grad_y(varB), mask_y)
        return loss
