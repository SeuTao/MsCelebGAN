import os
import argparse
from data_loader import get_pure_loader
from torch.backends import cudnn

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
from model import MultiResDecoder
import face_recognition_networks
from PIL import Image

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]



class DecoderSolver(object):

    def __init__(self, Msceleb_loader, config):
        # Data loader
        self.Msceleb_loader = Msceleb_loader

        # Model hyper-parameters
        self.c_dim = config.c_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.d_train_repeat = config.d_train_repeat

        # Hyper-parameteres
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        # self.dataset = config.dataset
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.batch_size = config.batch_size
        self.use_tensorboard = config.use_tensorboard
        self.pretrained_model = config.pretrained_model

        # Test settings
        self.test_model = config.test_model

        # Path
        self.log_path = config.log_path
        self.sample_path = config.sample_path
        self.model_save_path = config.model_save_path
        self.result_path = config.result_path

        # Step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step


        # Build tensorboard if use
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):

        self.G = MultiResDecoder()
        self.D_64 = Discriminator(self.d_conv_dim, 5)
        self.D_128 = Discriminator(self.d_conv_dim, 6)
        # self.C = Classifier(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)

        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_64_optimizer = torch.optim.Adam(self.D_64.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.d_128_optimizer = torch.optim.Adam(self.D_128.parameters(), self.d_lr, [self.beta1, self.beta2])
        # self.c_optimizer = torch.optim.Adam(self.C.parameters(), self.d_lr, [self.beta1, self.beta2])

        # Print networks
        self.print_network(self.G, 'G')
        self.print_network(self.D_64, 'D_64')
        self.print_network(self.D_128, 'D_128')
        # self.print_network(self.C, 'C')

        if torch.cuda.is_available():
            self.G.cuda()
            self.D_64.cuda()
            self.D_128.cuda()
            # self.C.cuda()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):

        if os.path.exists(os.path.join(self.model_save_path, '{}_G.pth'.format(self.pretrained_model))):
            self.G.load_state_dict(torch.load(os.path.join(
                self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))

            print('loaded trained G models (step: {})..!'.format(self.pretrained_model))

        if os.path.exists(os.path.join(self.model_save_path, '{}_D_64.pth'.format(self.pretrained_model))):
            self.D_64.load_state_dict(torch.load(os.path.join(
                self.model_save_path, '{}_D_64.pth'.format(self.pretrained_model))))

            print('loaded trained D_64 models (step: {})..!'.format(self.pretrained_model))

        if os.path.exists(os.path.join(self.model_save_path, '{}_D_128.pth'.format(self.pretrained_model))):
            self.D_128.load_state_dict(torch.load(os.path.join(
                self.model_save_path, '{}_D_128.pth'.format(self.pretrained_model))))

            print('loaded trained D_128 models (step: {})..!'.format(self.pretrained_model))

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_64_optimizer.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.d_128_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_64_optimizer.zero_grad()
        self.d_128_optimizer.zero_grad()


    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def threshold(self, x):
        x = x.clone()
        x[x >= 0.5] = 1
        x[x < 0.5] = 0
        return x


    def one_hot(self, labels, dim):
        """Convert label indices to one-hot vector"""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[torch.from_numpy(np.arange(batch_size).astype(np.int64)), labels.long()] = 1
        return out


    def train(self):
        """Train StarGAN within a single dataset."""
        self.criterionTV = TVLoss()
        self.data_loader = self.Msceleb_loader
        # The number of iterations per epoch
        iters_per_epoch = len(self.data_loader)

        fixed_x = []
        real_c = []
        for i, (images, _, inputs) in enumerate(self.data_loader):
            fixed_x.append(images)
            real_c.append(inputs)
            if i == 3:
                break

        # Fixed inputs and target domain labels for debugging
        fixed_c = torch.cat(real_c, dim=0)
        fixed_c = self.to_var(fixed_c, volatile=True)

        # lr cache for decaying
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start with trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[0])
        else:
            start = 0

        # Start training
        start_time = time.time()
        for e in range(start, self.num_epochs):
            for i, (images, images_64, inputs) in enumerate(self.data_loader):

                inputs = self.to_var(inputs)
                origin_x_64 = self.to_var(images_64)
                origin_x_128 = self.to_var(images)
                # # ================== Train D ================== #
                # Compute loss with real images
                out_src_64 = self.D_64(origin_x_64)
                out_src_128 = self.D_128(origin_x_128)
                d_loss_real_64 = - torch.mean(out_src_64)
                d_loss_real_128 = - torch.mean(out_src_128)


                # Compute loss with fake images
                fake_x_64, fake_x_128 = self.G(inputs)
                fake_x_64 = Variable(fake_x_64.data)
                fake_x_128 = Variable(fake_x_128.data)
                out_src_64 = self.D_64(fake_x_64)
                out_src_128 = self.D_128(fake_x_128)

                d_loss_fake_64 = torch.mean(out_src_64)
                d_loss_fake_128 = torch.mean(out_src_128)
                # Backward + Optimize
                d_loss_64 = d_loss_real_64 + d_loss_fake_64
                d_loss_128 = d_loss_real_128 + d_loss_fake_128

                self.reset_grad()

                d_loss_64.backward()
                self.d_64_optimizer.step()
                d_loss_128.backward()
                self.d_128_optimizer.step()

                # Compute gradient penalty
                def gradient_penalty(fake_x, origin_x, D, d_optimizer):

                    alpha = torch.rand(origin_x.size(0), 1, 1, 1).cuda().expand_as(origin_x)
                    interpolated = Variable(alpha * origin_x.data + (1 - alpha) * fake_x.data, requires_grad=True)
                    out = D(interpolated)

                    grad = torch.autograd.grad(outputs=out,
                                               inputs=interpolated,
                                               grad_outputs=torch.ones(out.size()).cuda(),
                                               retain_graph=True,
                                               create_graph=True,
                                               only_inputs=True)[0]

                    grad = grad.view(grad.size(0), -1)
                    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                    d_loss_gp = torch.mean((grad_l2norm - 1)**2)

                    # Backward + Optimize
                    d_loss = self.lambda_gp * d_loss_gp
                    self.reset_grad()
                    d_loss.backward()
                    d_optimizer.step()

                    return d_loss_gp

                d_loss_gp_64 = gradient_penalty(fake_x_64, origin_x_64, self.D_64, self.d_64_optimizer)
                d_loss_gp_128 = gradient_penalty(fake_x_128, origin_x_128, self.D_128, self.d_128_optimizer)

                # Logging
                loss = {}
                loss['D/loss_real'] = d_loss_real_64.data[0] + d_loss_real_128.data[0]
                loss['D/loss_fake'] = d_loss_fake_64.data[0] + d_loss_fake_128.data[0]
                loss['D/loss_gp'] = d_loss_gp_64.data[0] + d_loss_gp_128.data[0]

                # ================== Train G ================== #
                if (i+1) % self.d_train_repeat == 0:

                    # Original-to-target and target-to-original domain
                    fake_x_64, fake_x_128 = self.G(inputs)

                    # Compute losses
                    out_src_64 = self.D_64(fake_x_64)
                    out_src_128 = self.D_128(fake_x_128)

                    g_loss_fake_64 = - torch.mean(out_src_64)
                    g_loss_fake_128 = - torch.mean(out_src_128)

                    # Backward + Optimize
                    TV_loss = self.criterionTV(fake_x_64) * 0.001 + self.criterionTV(fake_x_128) * 0.001

                    g_loss = g_loss_fake_64 + g_loss_fake_128 + TV_loss

                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging
                    loss['G/loss_fake'] = g_loss_fake_64.data[0] + g_loss_fake_128.data[0]

                # Print out log info
                if (i+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed, e+1, self.num_epochs, i+1, iters_per_epoch)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)


                # Translate fixed images for debugging
                if (i+1) % self.sample_step == 0:
                    fake_64, fake_x_128 = self.G(fixed_c)

                    fake_images_64 = torch.cat([fake_x_64], dim=3)
                    fake_images_128 = torch.cat([fake_x_128], dim=3)
                    save_image(self.denorm(fake_images_64.data),
                        os.path.join(self.sample_path, '{}_{}_fake_64.png'.format(e+1, i+1)),nrow=1, padding=0)
                    save_image(self.denorm(fake_images_128.data),
                        os.path.join(self.sample_path, '{}_{}_fake_128.png'.format(e+1, i+1)),nrow=1, padding=0)
                    print('Translated images and saved into {}..!'.format(self.sample_path))

                # Save model checkpoints
                if (i+1) % self.model_save_step == 0:
                    torch.save(self.G.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_G.pth'.format(e+1, i+1)))
                    torch.save(self.D_64.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_D_64.pth'.format(e+1, i+1)))
                    torch.save(self.D_128.state_dict(),
                               os.path.join(self.model_save_path, '{}_{}_D_128.pth'.format(e + 1, i + 1)))



            # Decay learning rate
            if (e+1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


            torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_final_G.pth'.format(e + 1)))
            torch.save(self.D_64.state_dict(),
                           os.path.join(self.model_save_path, '{}_final_D_64.pth'.format(e + 1)))

            torch.save(self.D_128.state_dict(),
                       os.path.join(self.model_save_path, '{}_final_D_128.pth'.format(e + 1)))

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training
    cudnn.benchmark = True
    # Create directories if not exist
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    # Data loader
    MsCeleb_loader = get_pure_loader(config.image_path, config.metadata_path, config.crop_size,
                                   config.image_size, config.batch_size, 'CelebA', config.mode)

    # Solver
    solver = DecoderSolver(MsCeleb_loader, config)

    if config.mode == 'train':
        solver.train()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--c_dim', type=int, default=75628)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--g_repeat_num', type=int, default=6)
    parser.add_argument('--d_repeat_num', type=int, default=6)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    parser.add_argument('--lambda_cls', type=float, default=1)
    parser.add_argument('--lambda_rec', type=float, default=10)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--d_train_repeat', type=int, default=5)

    # Training settings
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_epochs_decay', type=int, default=10)
    parser.add_argument('--num_iters', type=int, default=200000)
    parser.add_argument('--num_iters_decay', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--pretrained_model', type=str, default=None)

    # Test settings
    parser.add_argument('--test_model', type=str, default='3_10000')

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Path
    parser.add_argument('--image_path', type=str, default=r'')
    parser.add_argument('--rafd_image_path', type=str, default=r'')

    parser.add_argument('--metadata_path', type=str, default=r'./MsCeleb_clean_aligned_75628.txt')

    parser.add_argument('--log_path', type=str, default='./pure_gan_multires/logs')
    parser.add_argument('--model_save_path', type=str, default='./pure_gan_multires/models')
    parser.add_argument('--sample_path', type=str, default='./pure_gan_multires/samples')
    parser.add_argument('--result_path', type=str, default='./pure_gan_multires/results')

    # Step size
    parser.add_argument('--log_step', type=int, default=200)
    parser.add_argument('--sample_step', type=int, default=5000)
    parser.add_argument('--model_save_step', type=int, default=20000)

    config = parser.parse_args()
    print(config)
    main(config)
