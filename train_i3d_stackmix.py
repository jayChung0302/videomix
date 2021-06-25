import os
import sys
import argparse
import time
import copy
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--mode_dir', type=str, help='rgb or opt dataset directory', default='C:/UCF101/tvl1_flow')
parser.add_argument('--split_path', type=str, help='', default='ucfTrainTestlist/')
parser.add_argument('--split', type=str, help='split way', default='01')
parser.add_argument('--root', type=str, help='frame count pickle directory', default='frame_count.pickle')
parser.add_argument('--init_lr', type=float, help='initial learning rate', default=0.1)
parser.add_argument('--mode', type=str, help='rgb or opt', default='opt')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--batchsize', type=int, help='batch size', default=4)
parser.add_argument('--save_model', type=str, default='ucf_finetuned')
parser.add_argument('--num_classes', type=int, help='number of classes', default=51)
parser.add_argument('--steps', type=float, help='number of steps', default=64e3)
parser.add_argument('--alpha', type=float, help='beta distribution hyper-parameter alpha', default=8.0)
parser.add_argument('--prob', type=float, help='probability to implement StackMix/TubeMix augmentation', default=0.5)

args = parser.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms
import numpy as np
import random

from i3d import InceptionI3d

from ucf_datasets import UCF101 as Dataset
# from hmdb_datasets import HMDB as Dataset
from mix import stackmix


def run(mode_dir, split_path, split, root, resume, init_lr=0.1, max_steps=args.steps, mode='opt',
        batch_size=8, save_model='ucf_fine', num_classes=101):
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
                                           ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    train_dataset = Dataset(mode_dir, split_path, split, stage='train', mode=mode,
                            pickle_dir=root, transforms=train_transforms)

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                             pin_memory=True)

    val_dataset = Dataset(mode_dir, split_path, split, stage='test', mode=mode, pickle_dir=root,
                          transforms=test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                                 pin_memory=True)

    dataloaders = {'train': dataloader, 'val': val_dataloader}

    if mode == 'rgb':
        net_dir = 'spatial/'
    else:
        net_dir = 'motion/'
    if not os.path.isdir('record/' + net_dir + args.save_model):
        os.mkdir('record/' + net_dir + args.save_model)
    logname = ('record/' + net_dir + args.save_model + '/' + args.save_model + '.csv')
    if not os.path.isdir(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['phase', 'epoch', 'localization loss', 'classification loss',
                                'total loss', 'epoch acc'])
    # setup the model
    if mode == 'opt':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)

    if resume:
        i3d.replace_logits(num_classes)
        if os.path.isfile(resume):
            print("==> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            i3d.load_state_dict(checkpoint)
            print('network loaded')
        else:
            print("==> no checkpoint found at '{}'".format(resume))
    else:
        if mode == 'opt':
            i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
        else:
            i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
        i3d.replace_logits(num_classes)

    lr = init_lr

    i3d = i3d.cuda()

    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=25, verbose=True,
                                                    threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0)

    num_steps_per_update = 4  # accumulate gradient
    steps = 0
    epoch = 1

    # train iteration
    while steps < max_steps:
        print('Step {}/{}'.format(steps, max_steps))
        print('epoch : {}'.format(epoch))
        print('-' * 10)

        for phase in ['train', 'val']:
            phase_time = time.time()
            if phase == 'train':
                i3d.train()
                epoch += 1
            else:
                i3d.eval()

            update_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0

            tot_corrects = 0
            tot_data = 0
            optimizer.zero_grad()

            for data in dataloaders[phase]:
                running_corrects = 0
                num_iter += 1
                with torch.set_grad_enabled(phase == 'train'):
                    inputs, labels = data
                    if phase == 'train':
                        inputs, labels, cls_labels = stackmix(inputs, labels, args.alpha, args.prob)
                        cls_labels = cls_labels.cuda()

                    inputs = inputs.cuda()
                    t = inputs.size(2)
                    labels = labels.cuda()

                    per_frame_logits = i3d(inputs)

                    # upsample to input size
                    per_frame_logits = nn.functional.interpolate(per_frame_logits, t, mode='linear', align_corners=True)

                    # compute localization loss
                    loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                    tot_loc_loss += loc_loss.item()

                    # compute classification loss (with max-pooling along time B x C x T)
                    if phase == 'train':
                        cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
                                                                      torch.max(cls_labels, dim=2)[0])
                    else:
                        cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
                                                                      torch.max(labels, dim=2)[0])
                    tot_cls_loss += cls_loss.item()

                    # compute classfication accuracy
                    _, preds = torch.max(torch.max(per_frame_logits, dim=2)[0], dim=1)
                    _, labels = torch.max(torch.max(labels, dim=2)[0], dim=1)

                    running_corrects += torch.sum(preds == labels.data)
                    tot_data += labels.size(0)

                    # optical flow 
                    # 4 (batch) x 2 (u,v) x 64 (frames) x 224 (width) x 224 (height)

                    if phase == 'train':
                        loss = 0.5 * loc_loss + 0.5 * cls_loss
                        loss.backward()
                    if phase == 'val':
                        loss = cls_loss
                    update_loss += loss.item()
                tot_corrects += running_corrects
                csv_loc = tot_loc_loss / (10 * num_steps_per_update)
                csv_cls = tot_cls_loss / (10 * num_steps_per_update)
                csv_tot = update_loss / 10
                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()

                    if steps % 64 == 0:
                        print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss / (
                                    10 * num_steps_per_update), tot_cls_loss / (10 * num_steps_per_update), update_loss / 10))

                        lr_sched.step(update_loss)
                        update_loss = tot_loc_loss = tot_cls_loss = 0.

            if phase == 'val':
                print('{}  Cls Loss: {:.4f} Tot Loss: {:.4f}'.
                      format(phase, tot_loc_loss / num_iter, (update_loss * num_steps_per_update) / num_iter))
                print('network prediction : {}'.format(preds))
                print('and the label is : {}'.format(labels))
            else:
                torch.save(i3d.state_dict(), 'record/' + net_dir + save_model + '/' + save_model + str(epoch-1).zfill(6) + '.pt')

            phase_end_time = time.time()
            epoch_acc = tot_corrects.item() / tot_data
            print(phase + ' accuaracy : {:.2f}% [{}/{}] phase time : {:.0f}m {:.0f}s'.format(epoch_acc * 100, tot_corrects.item(),
                                                                tot_data, (phase_end_time - phase_time)//60, (phase_end_time - phase_time) % 60))
            print()
            with open(logname, 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow([phase, epoch-1, csv_loc, csv_cls, csv_tot, epoch_acc])

if __name__ == '__main__':
    run(args.mode_dir, args.split_path, args.split, args.root, args.resume, args.init_lr, max_steps=args.steps,
        mode=args.mode,
        batch_size=args.batchsize, save_model=args.save_model, num_classes=args.num_classes)
