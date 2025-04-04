import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
from skimage import io
import torchvision
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import spectral_norm
from utils import lab_to_rgb
from classes import UnetBlock, UnetGenerator, PatchDiscriminatorSN, GANLoss

import torch
from torch import nn, optim

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def init_weights(net, init='kaiming'):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and ('Conv' in classname or 'Linear' in classname):
            if init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init == 'normal':
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)
    return net

def init_model(model):
    model = model.to(device)
    model = init_weights(model)
    return model

class MainModel(nn.Module):
    def __init__(self, net_G=None, net_D=None, lr_G=2e-4, lr_D=2e-4,
                 lambda_L1=100., lambda_perceptual=10.):
        super().__init__()
        self.device = device
        self.lambda_L1 = lambda_L1
        # self.lambda_perceptual = lambda_perceptual

        if net_G is None:
            self.net_G = init_model(UnetGenerator(input_c=1, output_c=2))
        else:
            self.net_G = net_G.to(self.device)

        if net_D is None:
            self.net_D = init_model(PatchDiscriminatorSN(input_c=3))
        else:
            self.net_D = net_D.to(self.device)

        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        # self.perceptual_loss = PerceptualLoss()

        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G,
                                betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D,
                                betas=(0.5, 0.999))

    def setup_input(self, data):
        self.L = data['L'].to(self.device)  # Input SAR image
        self.ab = data['ab'].to(self.device)  # Ground truth ab channels

    def forward(self):
        self.fake_color = self.net_G(self.L)  # Generate fake ab channels

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)  # Concatenate L and fake ab
        real_image = torch.cat([self.L, self.ab], dim=1)  # Concatenate L and real ab

        fake_preds = self.net_D(fake_image.detach())
        real_preds = self.net_D(real_image)

        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)

        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        # self.loss_G_perceptual = self.perceptual_loss(self.fake_color, self.ab) * self.lambda_perceptual

        self.loss_G = self.loss_G_GAN + self.loss_G_L1  # + self.loss_G_perceptual
        self.loss_G.backward()

    def optimize(self):
        # Update Discriminator
        self.forward()
        self.net_D.train()
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        # Update Generator
        self.net_G.train()
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()