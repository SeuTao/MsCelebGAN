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


def transfrom_img(img_name, save_path, attribute):
    img = Image.open(img_name).convert('L')
    # center crop
    w = img.size[0]
    h = img.size[1]
    h_times = int(h / 128)
    w_times = int(w / 128)

    if w_times>1:
        h_off = 0
        if attribute == 'glass':
            w_off = 1
        elif attribute == 'smile':
            w_off = 1+128
        elif attribute == 'aged':
            w_off = 1+256
        elif attribute == 'disgust':
            w_off = 1+384
        elif attribute == 'eye_closed':
            w_off = 1+512
        elif attribute == 'surprised':
            w_off = 1+640

        img1 = img.crop((w_off, h_off, w_off + 128, h_off + 128))
        # img1.show()
        img1.save(save_path)

path = r'/data6/shentao/MsCeleb_aligned_aug'
replace = r'MsCeleb_aligned'
attribute = 'eye_closed'
#
img_list = create_image_lists_recursive(path)
random.shuffle(img_list)

count = 0
for path_tmp in img_list:
    save_path = path_tmp.replace(replace+'_aug',replace+'_'+attribute)
    if not os.path.exists(os.path.split(save_path)[0]):
        os.makedirs(os.path.split(save_path)[0])

    if not os.path.exists(save_path):
        transfrom_img(path_tmp, save_path,attribute)
    print(save_path)
    count += 1
    print(count)



