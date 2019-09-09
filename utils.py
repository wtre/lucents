import matplotlib
import matplotlib.cm
import numpy as np

import torch
import torchvision.utils as vutils


def DepthNorm(depth, maxDepth=1000.0): 
    # return maxDepth / depth
    depth_n = maxDepth / depth
    z = torch.zeros(depth.size()).cuda()
    depth_n = torch.clamp(depth_n, 1, 1000)
    return torch.where(depth != 0, depth_n.to('cuda'), z)


def thresh_mask(depth_gt, depth_raw, thresh=3):
    dd = depth_raw - depth_gt
    z = torch.zeros(depth_raw.size()).cuda()
    o = torch.ones(depth_raw.size()).cuda()
    return torch.where(dd > thresh, z, o)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    value = value.cpu().numpy()[0,:,:]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    #value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)

    img = value[:,:,:3]

    return img.transpose((2,0,1))


def save_error_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Export Diverging image grid.
    Make sure range equals (-a, a)
    """
    from PIL import Image

    # Hand-craft diverging colormap (b-w-r)
    z = torch.zeros(tensor.size()).cuda()
    tensor_r = torch.where(tensor > 0, z, tensor)
    tensor_g = torch.where(tensor > 0, tensor.mul_(-1), tensor)
    tensor_b = torch.where(tensor > 0, tensor.mul_(-1), z)
    tensor_merged = torch.cat((tensor_r, tensor_g, tensor_b), 1)

    grid = vutils.make_grid(tensor_merged, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=(range[0], 0), scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)