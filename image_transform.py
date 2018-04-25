import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torch.autograd import grad
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
from model import Generator
from model import Discriminator
import face_recognition_networks
from PIL import Image
import glob
import random


def create_image_lists(image_dir, printable = True):
    if not os.path.exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None

    file_list = []
    file_glob = os.path.join(image_dir,'*.' + 'jpg')
    file_list.extend(glob.glob(file_glob))
    file_glob = os.path.join(image_dir,'*.' + 'png')
    file_list.extend(glob.glob(file_glob))
    file_glob = os.path.join(image_dir,'*.' + 'tif')
    file_list.extend(glob.glob(file_glob))
    file_glob = os.path.join(image_dir,'*.' + 'bmp')
    file_list.extend(glob.glob(file_glob))

    file_glob = os.path.join(image_dir,'*.' + 'raw')
    file_list.extend(glob.glob(file_glob))


    if printable:
        print(len(file_list))
    return file_list

def create_image_lists_recursive(image_dir):

    total_list = []
    for i in os.walk(image_dir):
        cur_path = i[0]
        list = create_image_lists(cur_path,printable=False)
        total_list.extend(list)

    print(len(total_list))
    return total_list


def denorm( x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def to_var( x, volatile=False):
    # if torch.cuda.is_available():
    x = x.cuda()
    return Variable(x, volatile=volatile)

def one_hot( labels, dim):
    """Convert label indices to one-hot vector"""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[torch.from_numpy(np.arange(batch_size).astype(np.int64)), labels.long()] = 1
    return out

model = Generator(64, 5+7+2, 6)   # 2 for mask vector
model.load_state_dict(torch.load(r'./stargan/facialmodels_7_add/200000_G.pth'))
model.cuda()
model.eval()

def transfrom_img(img_name, save_path, attribute):
    # CelebA = [0, 3, 4]
    # facial = [1, 2, 6]
    if attribute == 'glass':
        CelebA = [0]
        facial = []
    elif attribute == 'smile':
        CelebA = [3]
        facial = []
    elif attribute == 'aged':
        CelebA = [4]
        facial = []
    elif attribute == 'disgust':
        CelebA = []
        facial = [1]
    elif attribute == 'eye_closed':
        CelebA = []
        facial = [2]
    elif attribute == 'surprised':
        CelebA = []
        facial = [6]

    transform = transforms.Compose([transforms.CenterCrop(128),
                                    transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])
    count = 0
    c2_dim = 7
    c_dim = 5
    input = torch.zeros(1, 3, 128, 128)


    img = Image.open(img_name).convert('L').convert('RGB')
    # img   = np.reshape(img, (128, 128, 1))
    img = transform(img)
    input[0, :, :, :] = img
    # input = input.cuda()

    target_c1_list = []
    for j in range(c_dim):
        target_c = one_hot(torch.ones(input.size(0)) * j, c_dim)
        target_c1_list.append(to_var(target_c, volatile=True))

    target_c2_list = []
    for j in range(c2_dim):
        target_c = one_hot(torch.ones(input.size(0)) * j, c2_dim)
        target_c2_list.append(to_var(target_c, volatile=True))

    # Zero vectors and mask vectors
    zero1 = to_var(torch.zeros(input.size(0), c2_dim))  # zero vector for rafd expressions
    mask1 = to_var(one_hot(torch.zeros(input.size(0)), 2))  # mask vector: [1, 0]
    zero2 = to_var(torch.zeros(input.size(0), c_dim))  # zero vector for celebA attributes
    mask2 = to_var(one_hot(torch.ones(input.size(0)), 2))  # mask vector: [0, 1]
    input = to_var(input)

    # # Changing hair color, gender, and age
    fake_image_list = []
    for j in CelebA:
        target_c = torch.cat([target_c1_list[j], zero1, mask1], dim=1)
        fake_image_list.append(model(input, target_c))


    # Changing emotional expressions
    for j in facial:
        target_c = torch.cat([zero2, target_c2_list[j], mask2], dim=1)
        fake_image_list.append(model(input, target_c))

    fake_images = torch.cat(fake_image_list, dim=3)

    # Save the translated images

    save_image(denorm(fake_images.data), save_path, nrow=1, padding=0)

def transfrom_img_all(img_name, save_path):
    CelebA = [0, 3, 4]
    facial = [1, 2, 6]
    # if attribute == 'glass':
    #     CelebA = [0]
    #     facial = []
    # elif attribute == 'smile':
    #     CelebA = [3]
    #     facial = []
    # elif attribute == 'aged':
    #     CelebA = [4]
    #     facial = []
    # elif attribute == 'disgust':
    #     CelebA = []
    #     facial = [1]
    # elif attribute == 'eye_closed':
    #     CelebA = []
    #     facial = [2]
    # elif attribute == 'surprised':
    #     CelebA = []
    #     facial = [6]

    transform = transforms.Compose([transforms.CenterCrop(128),
                                    transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])
    count = 0
    c2_dim = 7
    c_dim = 5
    input = torch.zeros(1, 3, 128, 128)


    img = Image.open(img_name).convert('L').convert('RGB')
    # img   = np.reshape(img, (128, 128, 1))
    img = transform(img)
    input[0, :, :, :] = img
    # input = input.cuda()

    target_c1_list = []
    for j in range(c_dim):
        target_c = one_hot(torch.ones(input.size(0)) * j, c_dim)
        target_c1_list.append(to_var(target_c, volatile=True))

    target_c2_list = []
    for j in range(c2_dim):
        target_c = one_hot(torch.ones(input.size(0)) * j, c2_dim)
        target_c2_list.append(to_var(target_c, volatile=True))

    # Zero vectors and mask vectors
    zero1 = to_var(torch.zeros(input.size(0), c2_dim))  # zero vector for rafd expressions
    mask1 = to_var(one_hot(torch.zeros(input.size(0)), 2))  # mask vector: [1, 0]
    zero2 = to_var(torch.zeros(input.size(0), c_dim))  # zero vector for celebA attributes
    mask2 = to_var(one_hot(torch.ones(input.size(0)), 2))  # mask vector: [0, 1]
    input = to_var(input)

    # # Changing hair color, gender, and age
    fake_image_list = []
    for j in CelebA:
        target_c = torch.cat([target_c1_list[j], zero1, mask1], dim=1)
        fake_image_list.append(model(input, target_c))


    # Changing emotional expressions
    for j in facial:
        target_c = torch.cat([zero2, target_c2_list[j], mask2], dim=1)
        fake_image_list.append(model(input, target_c))

    fake_images = torch.cat(fake_image_list, dim=3)

    # Save the translated images

    save_image(denorm(fake_images.data), save_path, nrow=1, padding=0)

img_name = r'tmp3.jpg'
save_path = 'test3_eye.png'
attribute = 'eye_closed'

transfrom_img(img_name, save_path, attribute)

# path = r'/data6/shentao/MsCeleb_aligned'
# replace = r'MsCeleb_aligned'
# # attribute = 'aged'
#
# img_list = create_image_lists_recursive(path)
# random.shuffle(img_list)
#
# count = 0
# for path_tmp in img_list:
#     save_path = path_tmp.replace(replace,replace+'_aug')
#
#     if not os.path.exists(os.path.split(save_path)[0]):
#         os.makedirs(os.path.split(save_path)[0])
#
#     if not os.path.exists(save_path):
#         transfrom_img_all(path_tmp, save_path)
#     print(save_path)
#     count += 1
#     print(count)



