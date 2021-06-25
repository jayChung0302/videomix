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
    if prob < 0:
        raise ValueError('prob must be a positive value')

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

def speed_jitter(frames, alpha, speed_mode, mode='rgb', max_speed=5, prob=0.5):
    """range of key_idx is 2s+1<= k <= N-(2s+1)"""
    k = random.random()
    
    with torch.no_grad():
        if k > 0:
            if speed_mode == 'variation':
                speed = np.random.choice([i for i in range(max_speed + 1)])
    
            else:
                speed = int(speed_mode)
            lam = np.random.beta(alpha, alpha)
            key_idx = int((64-2*(2*speed+1))*lam + 2*speed + 1)
            batch_size = frames.size(0)
            if mode != 'rgb':
                new_group = torch.randn(batch_size, 2, (2*speed+1) * 2, 224, 224).cuda()
            else:
                new_group = torch.randn(batch_size, 3, (2*speed+1) * 2, 224, 224).cuda()
            for side in ['b', 'a']:
                for i in range(speed):
                    if side == 'b':
                        new_group[:, :, i*2, :, :] = frames[:, :, key_idx-speed-speed-1+i, :, :]
                        new_group[:, :, i*2 + 1, :, :] = frames[:, :, key_idx-speed-speed-1+i, :, :] * 0.5 + \
                                                         frames[:, :, key_idx-speed-speed-1+i+1, :, :] * 0.5
                    if side == 'a':
                        new_group[:, :, 2*speed + 1 + i*2, :, :] = frames[:, :, key_idx+speed+1+i, :, :]
                        new_group[:, :, 2*speed + 2 + i*2, :, :] = frames[:, :, key_idx+speed+1+i, :, :] * 0.5 + \
                                                                   frames[:, :, key_idx+speed+1+i+1, :, :] * 0.5
            new_group[:, :, 2*speed, :, :] = frames[:, :, key_idx-speed-1, :, :]
            new_group[:, :, 4*speed+1, :, :] = frames[:, :, key_idx+2*speed+1, :, :]
            new_frames = torch.cat((frames[:, :, :key_idx-2*speed-1, :, :], new_group[:, :, :2*speed + 1, :, :],
                                    frames[:, :, key_idx, :, :].unsqueeze(2), new_group[:, :, 2*speed+1:, :, :],
                                    frames[:, :, key_idx+2*speed+2:, :, :]), dim=2)
        else:
            return frames
    
        return new_frames

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
