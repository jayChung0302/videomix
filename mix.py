import torch
import numpy as np
import random

def stackmix(x, y, alpha, prob, nframes=64):
    if prob < 0:
        raise ValueError('prob must be a positive value')
    
    k = random.random()
    if k > 1 - prob:
        batch_size = x.size()[0]
        batch_idx = torch.randperm(batch_size)
        lam = np.random.beta(alpha, alpha)
        cut_idx = int(lam * nframes)
        shuffled_x = torch.cat((x[:, :, :cut_idx, :, :], x[batch_idx][:, :, cut_idx:, :, :]), dim=2)
        shuffled_y = torch.cat((y[:, :, :cut_idx], y[batch_idx][:, :, cut_idx:]), dim=2)
        cls_y = torch.cat((y[:, :, :cut_idx] * (cut_idx / nframes), y[batch_idx][:, :, cut_idx:] * (1 - cut_idx / nframes)), dim=2)
        return shuffled_x, shuffled_y, cls_y
    else:
        return x, y, y

def tubemix(x, y, alpha, prob):
    k = random.random()
    if k > 1 - prob:
        batch_size = x.size()[0]
        batch_idx = torch.randperm(batch_size)
        lam = np.random.beta(alpha, alpha)

        bbx1, bby1, bbx2, bby2 = rand_bbox(x[:, :, 0, :, :].size(), lam)
        x[:, :, :, bbx1:bbx2, bby1:bby2] = x[batch_idx, :, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        tube_y = y * lam + y[batch_idx] * (1 - lam)
        return x, tube_y
    else:
        return x, y

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
