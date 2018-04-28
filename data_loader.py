import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np


class MsCelebDataset(Dataset):
    def __init__(self, image_path, metadata_path, transform, mode):
        self.image_path = image_path
        self.transform = transform
        self.mode = mode

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


