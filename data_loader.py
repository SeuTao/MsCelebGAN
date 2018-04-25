import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np


class CelebDataset(Dataset):
    def __init__(self, image_path, metadata_path, transform, mode):
        self.image_path = image_path
        self.transform = transform
        self.mode = mode
        self.lines = open(metadata_path, 'r').readlines()
        self.num_data = int(self.lines[0])
        self.attr2idx = {}
        self.idx2attr = {}

        print ('Start preprocessing dataset..!')
        self.preprocess()
        print ('Finished preprocessing dataset..!')

        if self.mode == 'train':
            self.num_data = len(self.train_filenames)
        elif self.mode == 'test':
            self.num_data = len(self.test_filenames)

    def preprocess(self):
        attrs = self.lines[1].split()
        for i, attr in enumerate(attrs):
            self.attr2idx[attr] = i
            self.idx2attr[i] = attr

        # self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
        self.selected_attrs = ['Eyeglasses', 'Male', 'No_Beard', 'Smiling',  'Young']
        self.train_filenames = []
        self.train_labels = []
        self.test_filenames = []
        self.test_labels = []

        lines = self.lines[2:]
        random.shuffle(lines)   # random shuffling
        for i, line in enumerate(lines):

            splits = line.split()
            filename = splits[0]
            values = splits[1:]

            label = []
            for idx, value in enumerate(values):
                attr = self.idx2attr[idx]

                if attr in self.selected_attrs:
                    if value == '1':
                        label.append(1)
                    else:
                        label.append(0)

            if (i+1) < 2000:
                self.test_filenames.append(filename)
                self.test_labels.append(label)
            else:
                self.train_filenames.append(filename)
                self.train_labels.append(label)

    def __getitem__(self, index):
        if self.mode == 'train':
            image = Image.open(os.path.join(self.image_path, self.train_filenames[index])).convert("RGB")
            label = self.train_labels[index]
        elif self.mode in ['test']:
            image = Image.open(os.path.join(self.image_path, self.test_filenames[index])).convert("RGB")
            label = self.test_labels[index]

        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        return self.num_data



class MsCelebDataset(Dataset):
    def __init__(self, image_path, metadata_path, transform, mode):
        self.image_path = image_path
        self.transform = transform
        self.mode = mode
        # self.lines = open(metadata_path, 'r').readlines()

        def default_list_reader(fileList):
            imgList = []
            Aug_imgList = []

            count = 0
            with open(fileList, 'r') as file:
                for line in file.readlines():
                    line = line.strip()
                    pos = line.rfind(' ')
                    imgPath = line[:pos]
                    label = line[pos + 1:]

                    # print (label)
                    # print(imgPath)

                    # if os.path.exists(imgPath):
                    if r'Ms_3D_aligned' in imgPath:
                        Aug_imgList.append((imgPath, int(label)))
                    else:
                        imgList.append((imgPath, int(label)))

                    if count > 200000:
                        break

                    count += 1

                random.shuffle(imgList)
            return Aug_imgList, imgList

        self.Aug_imglist, self.Origin_list = default_list_reader(metadata_path)


        self.num_data = max(len(self.Aug_imglist), len(self.Origin_list))

        # if self.mode == 'train':
        #     self.num_data = len(self.train_filenames)
        # elif self.mode == 'test':
        #     self.num_data = len(self.test_filenames)

    def __getitem__(self, index):
        origin_index = index% len(self.Origin_list)
        origin_image = Image.open(os.path.join(self.image_path, self.Origin_list[origin_index][0])).convert("L")
        origin_label =  self.Origin_list[origin_index][1]

        aug_index = index% len(self.Aug_imglist)
        aug_image = Image.open(os.path.join(self.image_path, self.Aug_imglist[aug_index][0])).convert("L")
        aug_label =  self.Aug_imglist[aug_index][1]

        # tmp_0 = self.transform(aug_image)
        # tmp_1 = self.transform(origin_image)
        #
        # l0 = torch.from_numpy(np.asarray(aug_label).reshape([1]))
        # l1 = torch.from_numpy(np.asarray(origin_label).reshape([1]))

        # return self.transform(aug_image), torch.from_numpy(np.asarray(aug_label).reshape([-1,1])), \
        #        self.transform(origin_image), torch.from_numpy(np.asarray(origin_label).reshape([-1,1]))

        return self.transform(aug_image), aug_label, \
           self.transform(origin_image), origin_label

    def __len__(self):
        return self.num_data


class MsCelebDataset_pureGAN(Dataset):
    def __init__(self, image_path, metadata_path, transform, mode):
        self.image_path = image_path
        self.transform = transform
        self.mode = mode
        # self.lines = open(metadata_path, 'r').readlines()

        def default_list_reader(fileList):
            imgList = []
            # Aug_imgList = []

            count = 0
            with open(fileList, 'r') as file:
                for line in file.readlines():
                    line = line.strip()
                    pos = line.rfind(' ')
                    imgPath = line[:pos]
                    label = line[pos + 1:]


                    if not r'Ms_3D_aligned' in imgPath:
                        imgList.append((imgPath, int(label)))

                    # if count > 200000:
                    #     break
                    # count += 1

                random.shuffle(imgList)
            return imgList

        self.Origin_list = default_list_reader(metadata_path)
        self.num_data = len(self.Origin_list)


    def __getitem__(self, index):
        origin_index = index% len(self.Origin_list)
        origin_image = Image.open(os.path.join(self.image_path, self.Origin_list[origin_index][0])).convert("L")

        origin_image = origin_image.copy().resize((128, 128))
        origin_image_64 = origin_image.copy().resize((64,64))

        input = np.random.random(size=512).astype(np.float32)
        return self.transform(origin_image), self.transform(origin_image_64), input

    def __len__(self):
        return self.num_data

def pil_default_loader(path):
    img = Image.open(path).convert('RGB')
    return img

def get_loader(image_path, metadata_path, crop_size, image_size, batch_size, dataset='CelebA', mode='train'):
    """Build and return data loader."""

    if mode == 'train':
        transform = transforms.Compose([
            # transforms.CenterCrop(crop_size),
            # transforms.Scale(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            # transforms.CenterCrop(crop_size),
            # transforms.Scale(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = MsCelebDataset(image_path, metadata_path, transform, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def get_pure_loader(image_path, metadata_path, crop_size, image_size, batch_size, dataset='CelebA', mode='train'):
    """Build and return data loader."""

    if mode == 'train':
        transform = transforms.Compose([
            # transforms.CenterCrop(crop_size),
            # transforms.Scale(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            # transforms.CenterCrop(crop_size),
            # transforms.Scale(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = MsCelebDataset_pureGAN(image_path, metadata_path, transform, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


