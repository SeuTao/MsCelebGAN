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
from model import Classifier
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


class Solver(object):

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

        # self.lambda_face = 0.0
        # if self.lambda_face > 0.0:
        #     self.Face_recognition_network = face_recognition_networks.LightCNN_29Layers(num_classes=79077)
        #     self.Face_recognition_network = torch.nn.DataParallel(self.Face_recognition_network).cuda()
        #     checkpoint = torch.load(r'/data5/shentao/LightCNN/CNN_29.pkl')
        #     self.Face_recognition_network.load_state_dict(checkpoint)
        #     for param in self.Face_recognition_network.parameters():
        #         param.requires_grad = False
        #     self.Face_recognition_network.eval()

        # Build tensorboard if use
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):

        self.G = Generator(self.g_conv_dim, self.g_repeat_num)
        self.D = Discriminator(self.d_conv_dim, self.d_repeat_num)
        self.C = Classifier(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)

        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.c_optimizer = torch.optim.Adam(self.C.parameters(), self.d_lr, [self.beta1, self.beta2])

        # Print networks
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        self.print_network(self.C, 'C')

        if torch.cuda.is_available():
            self.G.cuda()
            self.D.cuda()
            self.C.cuda()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        self.C.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_C.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.c_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.c_optimizer.zero_grad()

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

    def compute_accuracy(self, x, y):
        _, predicted = torch.max(x, dim=1)
        correct = (predicted == y).float()
        accuracy = torch.mean(correct) * 100.0
        return accuracy

    def one_hot(self, labels, dim):
        """Convert label indices to one-hot vector"""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)

        out[torch.from_numpy(np.arange(batch_size).astype(np.int64)), labels.long()] = 1
        return out


    def train(self):
        """Train StarGAN within a single dataset."""
        self.criterionL1 = torch.nn.L1Loss()
        # self.criterionL2 = torch.nn.MSELoss()
        self.criterionTV = TVLoss()

        self.data_loader = self.Msceleb_loader
        # The number of iterations per epoch
        iters_per_epoch = len(self.data_loader)

        fixed_x = []
        real_c = []
        for i, (aug_images, aug_labels, _, _) in enumerate(self.data_loader):
            fixed_x.append(aug_images)
            real_c.append(aug_labels)
            if i == 3:
                break

        # Fixed inputs and target domain labels for debugging
        fixed_x = torch.cat(fixed_x, dim=0)
        fixed_x = self.to_var(fixed_x, volatile=True)

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
            for i, (aug_x, aug_label, origin_x, origin_label) in enumerate(self.data_loader):

                # Generat fake labels randomly (target domain labels)
                # aug_c = self.one_hot(aug_label, self.c_dim)
                # origin_c = self.one_hot(origin_label, self.c_dim)

                aug_c_V = self.to_var(aug_label)
                origin_c_V = self.to_var(origin_label)

                aug_x = self.to_var(aug_x)
                origin_x = self.to_var(origin_x)

                # # ================== Train D ================== #
                # Compute loss with real images
                out_src = self.D(origin_x)
                out_cls = self.C(origin_x)
                d_loss_real = - torch.mean(out_src)

                c_loss_cls = F.cross_entropy(out_cls, origin_c_V)
                # Compute classification accuracy of the discriminator
                if (i+1) % self.log_step == 0:
                    accuracies = self.compute_accuracy(out_cls, origin_c_V)
                    log = ["{:.2f}".format(acc) for acc in accuracies.data.cpu().numpy()]
                    print('Classification Acc (75268 ids): ')
                    print(log)

                # Compute loss with fake images
                fake_x = self.G(aug_x)
                fake_x = Variable(fake_x.data)
                out_src = self.D(fake_x)
                d_loss_fake = torch.mean(out_src)

                # Backward + Optimize
                d_loss = d_loss_real + d_loss_fake
                c_loss = self.lambda_cls * c_loss_cls


                self.reset_grad()
                d_loss.backward()
                c_loss.backward()
                self.d_optimizer.step()
                self.c_optimizer.step()

                # Compute gradient penalty
                alpha = torch.rand(origin_x.size(0), 1, 1, 1).cuda().expand_as(origin_x)
                interpolated = Variable(alpha * origin_x.data + (1 - alpha) * fake_x.data, requires_grad=True)
                out = self.D(interpolated)

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
                self.d_optimizer.step()

                # Logging
                loss = {}
                loss['D/loss_real'] = d_loss_real.data[0]
                loss['D/loss_fake'] = d_loss_fake.data[0]
                loss['D/loss_gp'] = d_loss_gp.data[0]
                loss['C/loss_cls'] = c_loss_cls.data[0]

                # ================== Train G ================== #
                if (i+1) % self.d_train_repeat == 0:

                    # Original-to-target and target-to-original domain
                    fake_x = self.G(aug_x)

                    # Compute losses
                    out_src = self.D(fake_x)
                    out_cls = self.C(fake_x)
                    g_loss_fake = - torch.mean(out_src)

                    g_loss_cls = F.cross_entropy(out_cls, aug_c_V)

                    # Backward + Optimize
                    recon_loss = self.criterionL1(fake_x, aug_x)
                    TV_loss = self.criterionTV(fake_x) * 0.001

                    g_loss = g_loss_fake + self.lambda_cls * g_loss_cls + 5* recon_loss + TV_loss

                    # if self.lambda_face > 0.0:
                    #     self.criterionFace = nn.L1Loss()
                    #
                    #     real_input_x = (torch.sum(real_x, 1, keepdim=True) / 3.0 + 1) / 2.0
                    #     fake_input_x = (torch.sum(fake_x, 1, keepdim=True) / 3.0 + 1) / 2.0
                    #     rec_input_x = (torch.sum(rec_x, 1, keepdim=True) / 3.0 + 1) / 2.0
                    #
                    #     _, real_x_feature_fc, real_x_feature_conv = self.Face_recognition_network.forward(
                    #         real_input_x)
                    #     _, fake_x_feature_fc, fake_x_feature_conv = self.Face_recognition_network.forward(
                    #         fake_input_x)
                    #     _, rec_x1_feature_fc, rec_x1_feature_conv = self.Face_recognition_network.forward(rec_input_x)
                    #     # x1_loss = (self.criterionFace(fake_x1_feature_fc, Variable(real_x1_feature_fc.data,requires_grad=False)) +
                    #     #            self.criterionFace(fake_x1_feature_conv,Variable(real_x1_feature_conv.data,requires_grad=False)))\
                    #     #            * self.lambda_face
                    #     x_loss = (self.criterionFace(fake_x_feature_fc,Variable(real_x_feature_fc.data, requires_grad=False))) \
                    #               * self.lambda_face
                    #
                    #     rec_x_loss = (self.criterionFace(rec_x1_feature_fc, Variable(real_x_feature_fc.data, requires_grad=False)))
                    #
                    #     self.id_loss = x_loss + rec_x_loss
                    #     loss['G/id_loss'] = self.id_loss.data[0]
                    #     g_loss += self.id_loss

                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging
                    loss['G/loss_fake'] = g_loss_fake.data[0]
                    loss['G/loss_cls'] = g_loss_cls.data[0]

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
                    fake_image_list = [fixed_x]

                    fake_image_list.append(self.G(fixed_x))

                    fake_images = torch.cat(fake_image_list, dim=3)
                    save_image(self.denorm(fake_images.data),
                        os.path.join(self.sample_path, '{}_{}_fake.png'.format(e+1, i+1)),nrow=1, padding=0)
                    print('Translated images and saved into {}..!'.format(self.sample_path))

                # Save model checkpoints
                if (i+1) % self.model_save_step == 0:
                    torch.save(self.G.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_G.pth'.format(e+1, i+1)))
                    torch.save(self.D.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_D.pth'.format(e+1, i+1)))
                    torch.save(self.C.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_C.pth'.format(e+1, i+1)))


            # Decay learning rate
            if (e+1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


            torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_final_G.pth'.format(e + 1)))
            torch.save(self.D.state_dict(),
                           os.path.join(self.model_save_path, '{}_final_D.pth'.format(e + 1)))
            torch.save(self.C.state_dict(),
                           os.path.join(self.model_save_path, '{}_final_C.pth'.format(e + 1)))


    # def test(self):
    #     """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
    #     # Load trained parameters
    #     G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(self.test_model))
    #     self.G.load_state_dict(torch.load(G_path))
    #     self.G.eval()
    #
    #     if self.dataset == 'CelebA':
    #         data_loader = self.celebA_loader
    #     else:
    #         data_loader = self.rafd_loader
    #
    #     for i, (real_x, org_c) in enumerate(data_loader):
    #         real_x = self.to_var(real_x, volatile=True)
    #
    #         if self.dataset == 'CelebA':
    #             target_c_list = self.make_celeb_labels(org_c)
    #         else:
    #             target_c_list = []
    #             for j in range(self.c_dim):
    #                 target_c = self.one_hot(torch.ones(real_x.size(0)) * j, self.c_dim)
    #                 target_c_list.append(self.to_var(target_c, volatile=True))
    #
    #         # Start translations
    #         fake_image_list = [real_x]
    #         for target_c in target_c_list:
    #             fake_image_list.append(self.G(real_x, target_c))
    #         fake_images = torch.cat(fake_image_list, dim=3)
    #         save_path = os.path.join(self.result_path, '{}_fake.png'.format(i+1))
    #         save_image(self.denorm(fake_images.data), save_path, nrow=1, padding=0)
    #         print('Translated test images and saved into "{}"..!'.format(save_path))
