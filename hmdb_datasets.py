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
import glob


class HMDB_splitter():
    def __init__(self, path, split, stage, save_action_label=False):
        self.path = path
        self.split = split
        self.stage = stage
        self.save_action_label = save_action_label

    def get_action_index(self):
        self.action_label = {}
        self.action_idx = []
        self.train_video = {}
        self.val_video = {}
        self.test_video = {}
        dirs = glob.glob(os.path.join(self.path, '*.txt'))
        idx = 0
        for line in dirs:
            strs = line.split('\\')[1].split('_test_')
            action, split_ver = strs[0], strs[1]

            if action not in self.action_label.keys():
                self.action_label[action] = idx
                idx += 1
            if split_ver == 'split' + self.split + '.txt':
                self.action_idx.append(line)
        if self.save_action_label:
            with open('action_label_hmdb51.pickle', 'w') as f:
                pickle.dump(self.action_idx)  # there is some glob function error because of the folder name with brackets.
            f.close()


    def split_video(self):
        self.get_action_index()
        for videonamestxt in self.action_idx:
            label = str(videonamestxt).split('_test_')[0].split('\\')[1]
            with open(videonamestxt) as f:
                content = f.readlines()
            f.close()
            for i in content:
                video, mode = i.split(' ')[0], i.split(' ')[1]
                if mode == '1':
                    self.train_video[video] = self.action_label[label]
                elif mode == '2':
                    self.test_video[video] = self.action_label[label]
                elif mode == '0':
                    self.val_video[video] = self.action_label[label]
        print('>>> split mode : {}'.format(self.split))
        print('>>> (training video, validation video, test video) : {}, {}, {}'.format(len(self.train_video),
                                                                                       len(self.val_video),
                                                                                       len(self.test_video)))
        #         self.train_video = self.name_HandstandPushups(train_video)
        #         self.test_video = self.name_HandstandPushups(test_video)
        if self.stage == 'train':
            return self.train_video
        elif self.stage == 'test':
            return self.test_video
        elif self.stage == 'val':
            return self.val_video
        else:
            print('There are two options : train, test, val')


def load_rgb_frames(image_dir, vid, start, num, total_frames):
    vid = vid.split('.avi')[0]
    frames = []

    for i in range(start, start + num):
        j = i
        while j > total_frames:
            j = j - total_frames

        # try:
        img = cv2.imread(os.path.join(image_dir, vid, 'frame' + str(j).zfill(6) + '.jpg'))[:, :, [2, 1, 0]]
        w, h, c = img.shape

        # except AttributeError:
        #     print('image is not on set, dir: ', os.path.join(image_dir, 'u', vid, 'frame' + str(j).zfill(6) + '.jpg'))

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
    vid = vid.split('.avi')[0]
    total_frames = total_frames-1  # The optical flow database of hmdb-51 is one less than rgb.
    frames = []

    for i in range(start, start + num):
        j = i
        while j > total_frames:
            j = j - total_frames
        try:
            imgx = cv2.imread(os.path.join(image_dir, 'u', vid, 'frame' +
                                           str(j).zfill(6) + '.jpg'), cv2.IMREAD_GRAYSCALE)
            imgy = cv2.imread(os.path.join(image_dir, 'v', vid, 'frame' +
                                           str(j).zfill(6) + '.jpg'), cv2.IMREAD_GRAYSCALE)

            w, h = imgx.shape

        except TypeError:
            print('image is not on set, dir: ', os.path.join(image_dir, 'u', vid, 'frame' + str(j).zfill(6) + '.jpg'))
            sys.exit()
        except AttributeError:
            print('image is not on set, dir: ', os.path.join(image_dir, 'u', vid, 'frame' + str(j).zfill(6) + '.jpg'))
            sys.exit()

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


def make_dataset(root, data, num_classes=51):
    # root : num of frames pickle -dir
    # data : dictionary of hmdb-51 folder (from HMDB splitter.split_video)
    # mode : 'opt' or 'rgb' - deleted

    dataset = []
    with open(root, 'rb') as file:
        dic_frame = pickle.load(file)
    file.close()

    for vid in list(data.keys()):
        num_frames = dic_frame[vid.split('.')[0]]
        label = data[vid]
        labels = np.zeros((num_classes, 64), np.float32)  # 101, frame) size label generation
        labels[label, :] = 1
        dataset.append((vid, labels, num_frames))
    return dataset


class HMDB(data_utl.Dataset):
    def __init__(self, data_dir, split_path, split_mode, stage='train',
                 mode='rgb', pickle_dir='', transforms=None):

        self.data_dir = data_dir
        self.split_path = split_path
        self.split_mode = split_mode
        self.stage = stage
        self.mode = mode
        self.pickle_dir = pickle_dir
        self.transforms = transforms

        splitter = HMDB_splitter(path=split_path, split=split_mode, stage=stage)
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
            imgs = load_rgb_frames(self.data_dir, vid, start_f, 64, nf)
        else:
            imgs = load_flow_frames(self.data_dir, vid, start_f, 64, nf)


        imgs = self.transforms(imgs)
        label = torch.from_numpy(label)
        # return torch.from_numpy(label)
        # return video_to_tensor(imgs), label, 'v_' + vid
        return video_to_tensor(imgs), label

    def __len__(self):
        return len(self.data)


def video_imshow(inp, labels, mode):
    """Imshow for 5 dim Tensor."""
    if mode == 'rgb':
        fig1 = plt.figure(figsize=(9, 9))
        img = inp.squeeze().numpy().transpose((2, 3, 0, 1))
        rows = 8
        cols = 8
        i = 1
        for idx in range(img.shape[3]):
            ax1 = fig1.add_subplot(rows, cols, i)
            ax1.imshow(((img[:, :, :, idx]+1)*255/2).astype(np.uint8))

            ax1.set_xlabel(str(i), size=6)
            ax1.set_xticks([])
            ax1.set_yticks([])

            i = i + 1
        plt.show()

    else:
        u = inp.squeeze().numpy()[0, :, :, :].squeeze().transpose((1, 2, 0))
        v = inp.squeeze().numpy()[1, :, :, :].squeeze().transpose((1, 2, 0))
        fig1 = plt.figure(figsize=(9,9))
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
    split_path = 'HMDB-51_splits/'
    split = '1'
    root = 'hmdb_frame_counts.pickle'
    rgb_dir = 'E:/Dropbox/LAB/DATASET/hmdb-51/jpegs_256'
    opt_dir = 'E:/Dropbox/LAB/DATASET/hmdb-51/tvl1_flow'
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(), ])
    train_dataset = HMDB(opt_dir, split_path, split, stage='train', mode='opt',
                           pickle_dir=root, transforms=train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1,
                                             pin_memory=True)
    for data in train_dataloader:
        x, Y = data
        print(x.size(), Y.size())
        break

    # plot examples of opt mode
    inputs, labels = next(iter(train_dataloader))
    print(inputs.size())
    video_imshow(inputs, labels, mode='opt')
