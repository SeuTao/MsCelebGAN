import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)



class Generator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        # replicate spatially and concatenate domain information
        return self.main(x)



class Encoder(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(64, eps=0.001)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(64, eps=0.001)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(128, eps=0.001)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(128, eps=0.001)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(256, eps=0.001)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(256, eps=0.001)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(256, eps=0.001)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(512, eps=0.001)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(512, eps=0.001)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(512, eps=0.001)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(512, eps=0.001)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(512, eps=0.001)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(512, eps=0.001)

    def forward(self, X):

        h = F.relu(self.conv1_1_bn(self.conv1_1(X)))
        h = F.relu(self.conv1_2_bn(self.conv1_2(h)))

        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2) #64

        h = F.relu(self.conv2_1_bn(self.conv2_1(h)))
        h = F.relu(self.conv2_2_bn(self.conv2_2(h)))
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2) #32

        h = F.relu(self.conv3_1_bn(self.conv3_1(h)))
        h = F.relu(self.conv3_2_bn(self.conv3_2(h)))
        h = F.relu(self.conv3_3_bn(self.conv3_3(h)))
        # relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2) #16

        h = F.relu(self.conv4_1_bn(self.conv4_1(h)))
        h = F.relu(self.conv4_2_bn(self.conv4_2(h)))
        h = F.relu(self.conv4_3_bn(self.conv4_3(h)))

        h = F.max_pool2d(h, kernel_size=2, stride=2) #8

        h = F.relu(self.conv5_1_bn(self.conv5_1(h)))
        h = F.relu(self.conv5_2_bn(self.conv5_2(h)))
        h = F.relu(self.conv5_3_bn(self.conv5_3(h)))

        h = F.avg_pool2d(h, kernel_size=8, stride=8) # 512d vector

        return h



class Decoder(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv5_2 = nn.Conv2d(8, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(512, eps=0.001)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(512, eps=0.001)

        self.conv4_1_upsample = nn.Upsample(scale_factor=2)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(512, eps=0.001)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(512, eps=0.001)

        self.conv3_1_upsample = nn.Upsample(scale_factor=2)
        self.conv3_2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(256, eps=0.001)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(256, eps=0.001)

        self.conv2_1_upsample = nn.Upsample(scale_factor=2)
        self.conv2_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(128, eps=0.001)

        self.conv1_1_upsample = nn.Upsample(scale_factor=2)
        self.conv1_1 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)


    def forward(self, X):

        h = X.view(-1,8,8,8)

        h = F.relu(self.conv5_2_bn(self.conv5_2(h)))
        h = F.relu(self.conv5_3_bn(self.conv5_3(h)))

        h = self.conv4_1_upsample(h)
        h = F.relu(self.conv4_2_bn(self.conv4_2(h)))
        h = F.relu(self.conv4_3_bn(self.conv4_3(h)))

        h = self.conv3_1_upsample(h)
        h = F.relu(self.conv3_2_bn(self.conv3_2(h)))
        h = F.relu(self.conv3_3_bn(self.conv3_3(h)))

        h = self.conv2_1_upsample(h)
        h = F.relu(self.conv2_2_bn(self.conv2_2(h)))

        h = self.conv1_1_upsample(h)
        h = F.tanh(self.conv1_1(h))

        return h



class MultiResDecoder(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self):
        super(MultiResDecoder, self).__init__()

        self.conv5_2 = nn.Conv2d(8, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(512, eps=0.001)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(512, eps=0.001)

        self.conv4_1_upsample = nn.Upsample(scale_factor=2)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(512, eps=0.001)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(512, eps=0.001)

        self.conv3_1_upsample = nn.Upsample(scale_factor=2)
        self.conv3_2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(256, eps=0.001)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(256, eps=0.001)

        self.conv2_1_upsample = nn.Upsample(scale_factor=2)
        self.conv2_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(128, eps=0.001)

        self.conv_64 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

        self.conv1_1_upsample = nn.Upsample(scale_factor=2)
        self.conv1_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(128, eps=0.001)

        self.conv_128 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)


    def forward(self, X):

        h = X.view(-1,8,8,8)

        h = F.relu(self.conv5_2_bn(self.conv5_2(h)))
        h = F.relu(self.conv5_3_bn(self.conv5_3(h)))

        h = self.conv4_1_upsample(h)
        h = F.relu(self.conv4_2_bn(self.conv4_2(h)))
        h = F.relu(self.conv4_3_bn(self.conv4_3(h)))

        h = self.conv3_1_upsample(h)
        h = F.relu(self.conv3_2_bn(self.conv3_2(h)))
        h = F.relu(self.conv3_3_bn(self.conv3_3(h)))

        h = self.conv2_1_upsample(h)
        h = F.relu(self.conv2_2_bn(self.conv2_2(h)))

        re_64 = F.tanh(self.conv_64(h))

        h = self.conv1_1_upsample(h)
        h = F.relu(self.conv1_1_bn(self.conv1_1(h)))

        re_128 = F.tanh(self.conv_128(h))

        return re_64, re_128

class Discriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, conv_dim=64, repeat_num=6):
        super(Discriminator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(conv_dim, eps=0.001))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(curr_dim*2, eps=0.001))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_real = self.conv1(h)
        return out_real.squeeze()



class Classifier(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Classifier, self).__init__()

        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(conv_dim, eps=0.001))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(curr_dim*2, eps=0.001))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_aux = self.conv2(h)
        return out_aux.squeeze()


