import torch
import torch.utils.data as data_utl
import torchvision
import torchvision.models as models
import torch.utils.data as data_utl
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import os
import random
from torchvision import datasets, transforms
import videotransforms
import visdom
import sys

class UCF101_splitter():
    def __init__(self, path, split, stage):
        self.path = path
        self.split = split
        self.stage = stage

    def get_action_index(self):
        self.action_label = {}
        with open(self.path + 'classInd.txt') as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        for line in content:
            label, action = line.split(' ')
            if action not in self.action_label.keys():
                self.action_label[action] = label

    def split_video(self):
        self.get_action_index()
        for path, subdir, files in os.walk(self.path):
            for filename in files:
                if filename.split('.')[0] == 'trainlist' + self.split:
                    train_video = self.file2_dic(self.path + filename)
                if filename.split('.')[0] == 'testlist' + self.split:
                    test_video = self.file2_dic(self.path + filename)
        print('==> (Training video, Validation video):(', len(train_video), len(test_video), ')')
        self.train_video = self.name_HandstandPushups(train_video)
        self.test_video = self.name_HandstandPushups(test_video)
        if self.stage == 'train':
            return self.train_video
        elif self.stage == 'test':
            return self.test_video
        else:
            print('There are two options : train, test')

    def file2_dic(self, fname):
        with open(fname) as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        dic = {}
        for line in content:
            video = line.split('/', 1)[1].split(' ', 1)[0]
            key = video.split('_', 1)[1].split('.', 1)[0]
            label = self.action_label[line.split('/')[0]]
            dic[key] = int(label)
        return dic

    def name_HandstandPushups(self, dic):
        dic2 = {}
        for video in dic:
            n, g = video.split('_', 1)
            if n == 'HandStandPushups':
                videoname = 'HandstandPushups_' + g
            else:
                videoname = video
            dic2[videoname] = dic[video]
        return dic2


def load_rgb_frames(image_dir, vid, start, num, total_frames):
    frames = []

    for i in range(start, start + num):
        j = i
        while j > total_frames:
            j = j - total_frames

        try:
            img = cv2.imread(os.path.join(image_dir, vid, 'frame' + str(j).zfill(6) + '.jpg'))[:, :, [2, 1, 0]]
            w, h, c = img.shape

        # except AttributeError:
        except TypeError:
            print('image is not on set, dir: ', os.path.join(image_dir, 'u', vid, 'frame' + str(j).zfill(6) + '.jpg'))

        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def load_flow_frames(image_dir, vid, start, num, total_frames):
    # image_dir : 'C:/UCF101/tvl1_flow'
    # vid : 'v_ApplyEyeMakeup_g03_c01' ~
    # start : starting point index
    # num : number of video frames to get
    # total_frames : length of that video clips
    frames = []

    for i in range(start, start + num):
        j = i
        while j > total_frames:
            j = j - total_frames

        # try:
        imgx = cv2.imread(os.path.join(image_dir, 'u', vid, 'frame' +
                                       str(j).zfill(6) + '.jpg'), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(image_dir, 'v', vid, 'frame' +
                                       str(j).zfill(6) + '.jpg'), cv2.IMREAD_GRAYSCALE)

        w, h = imgx.shape

        # except AttributeError:
        #     print('image is not on set, dir: ', os.path.join(image_dir, 'u', vid, 'frame' + str(j).zfill(6) + '.jpg'))

        if w < 224 or h < 224:
            d = 224. - min(w, h)
            sc = 1 + d / min(w, h)
            imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
            imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

        imgx = (imgx / 255.) * 2 - 1
        imgy = (imgy / 255.) * 2 - 1
        img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def make_dataset(root, data, num_classes=101):
    # root : num of frames pickle -dir
    # data : dictionary of UCF-101 folder (from UCF101 splitter.split_video)
    # mode : 'opt' or 'rgb' - deleted

    dataset = []
    i = 0
    with open(root, 'rb') as file:
        dic_frame = pickle.load(file)
    file.close()
    # if mode == 'rgb':
    #     s = ''
    # elif mode == 'opt':
    #     s = 'v_'
    for vid in list(data.keys()):
        num_frames = dic_frame['v_' + vid + '.avi']
        label = data[vid]
        labels = np.zeros((num_classes, 64), np.float32)  # 101, frame) size label generation
        labels[label-1, :] = 1
        dataset.append((vid, labels, num_frames))
        i += 1
    return dataset


class UCF101(data_utl.Dataset):
    def __init__(self, data_dir, split_path, split_mode, stage='train',
                 mode='opt', pickle_dir='', transforms=None):

        self.data_dir = data_dir
        self.split_path = split_path
        self.split_mode = split_mode
        self.stage = stage
        self.mode = mode
        self.pickle_dir = pickle_dir
        self.transforms = transforms

        splitter = UCF101_splitter(path=split_path, split=split_mode, stage=stage)
        self.database = splitter.split_video()
        self.data = make_dataset(pickle_dir, self.database)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, nf = self.data[index]
        if nf <= 64:
            start_f = random.randint(1, nf)
        else:
            start_f = random.randint(1, nf-64)

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.data_dir, 'v_' + vid, start_f, 64, nf)
        else:
            imgs = load_flow_frames(self.data_dir, 'v_' + vid, start_f, 64, nf)

        # label = label[:, start_f:start_f + 64]

        imgs = self.transforms(imgs)
        label = torch.from_numpy(label)
        # return torch.from_numpy(label)
        return video_to_tensor(imgs), label
        # return video_to_tensor(imgs), label, 'v_' + vid
        #todo: 위에 'v_'+ vid 는 test 할시에만. 훈련할땐 제거해주자.

    def __len__(self):
        return len(self.data)


def video_imshow(inp, labels, classes, mode):
    """Imshow for 5 dim Tensor."""
    if mode == 'rgb':
        fig1 = plt.figure(figsize=(9, 9))
        # 1,3,64,224,224
        img = inp.squeeze().numpy().transpose((2, 3, 0, 1))
        rows = 8
        cols = 8
        i = 1
        for idx in range(img.shape[3]):
            ax1 = fig1.add_subplot(rows, cols, i)
            ax1.imshow(img[:, :, :, idx])
            ax1.set_xlabel(str(i), size=6)
            ax1.set_xticks([])
            ax1.set_yticks([])

            i = i + 1
        plt.show()

    else:
        # inp = inp.squeeze()[0,0,:,:].numpy().transpose((1, 2, 0))
        u = inp.squeeze().numpy()[0, :, :, :].squeeze().transpose((1, 2, 0))
        v = inp.squeeze().numpy()[1, :, :, :].squeeze().transpose((1, 2, 0))
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # inp = std * inp + mean
        # inp = np.clip
        fig1 = plt.figure(figsize=(9,9))
        # print((classes[labels.item()].split()[1]))
        fig2 = plt.figure(figsize=(9,9))

        rows = 8
        cols = 8
        i = 1
        for idx in range(u.shape[2]):
            ax1 = fig1.add_subplot(rows, cols, i)
            ax1.imshow(u[:, :, idx], cmap='gray')
            ax1.set_xlabel(str(i), size=6)
            ax1.set_xticks([])
            ax1.set_yticks([])
            # ax1.set_title(classes[labels.item()].split()[1])

            ax2 = fig2.add_subplot(rows, cols, i)
            ax2.imshow(v[:, :, idx], cmap='gray')
            ax2.set_xlabel(str(i), size=6)
            ax2.set_xticks([])
            ax2.set_yticks([])
            # ax2.set_title(classes[labels.item()].split()[1])

            i = i + 1
        plt.show()


if __name__ == '__main__':
    split_path = 'ucfTrainTestlist/'
    split = '02'
    root = 'frame_count.pickle'
    rgb_dir = 'C:/DATASET/jpegs_256'
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(), ])
    train_dataset = UCF101(rgb_dir, split_path, split, stage='train', mode='rgb',
                           pickle_dir=root, transforms=train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4,
                                             pin_memory=True)
    for data in train_dataloader:
        x, Y = data
        print(x.size(), Y)
        break

    # class names
    with open('classInd.txt') as f:
        class_names = f.readlines()
        class_names = [x.strip('\r\n') for x in class_names]
    f.close()

    # plot examples of opt mode
    inputs, labels = next(iter(train_dataloader))
    print(inputs.size())
    video_imshow(inputs, labels, class_names, mode='rgb')
