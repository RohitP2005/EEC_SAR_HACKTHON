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

import torch
from torch import nn, optim

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class UnetBlock(nn.Module):
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False):
        super().__init__()
        self.outermost = outermost
        if input_c is None:
            input_c = nf
        downconv = nn.Conv2d(input_c, ni, kernel_size=4, stride=2,
                             padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)

        if outermost:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(ni, nf, kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout:
                up += [nn.Dropout(0.5)]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
        
class UnetGenerator(nn.Module):
    def __init__(self, input_c=1, output_c=2, num_downs=8, num_filters=64):
        super().__init__()
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True)
        for _ in range(num_downs - 5):
            unet_block = UnetBlock(num_filters * 8, num_filters * 8,
                                   submodule=unet_block, dropout=True)
        unet_block = UnetBlock(num_filters * 4, num_filters * 8,
                               submodule=unet_block)
        unet_block = UnetBlock(num_filters * 2, num_filters * 4,
                               submodule=unet_block)
        unet_block = UnetBlock(num_filters, num_filters * 2,
                               submodule=unet_block)
        self.model = UnetBlock(output_c, num_filters, input_c=input_c,
                               submodule=unet_block, outermost=True)

    def forward(self, x):
        return self.model(x)
    

class PatchDiscriminatorSN(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        layers = [self.get_layers(input_c, num_filters, norm=False)]
        nf_mult = 1
        for n in range(1, n_down):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [self.get_layers(num_filters * nf_mult_prev,
                                       num_filters * nf_mult, s=2)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_down, 8)
        layers += [self.get_layers(num_filters * nf_mult_prev,
                                   num_filters * nf_mult, s=1)]
        layers += [spectral_norm(nn.Conv2d(num_filters * nf_mult, 1,
                                           kernel_size=4, stride=1, padding=1))]
        self.model = nn.Sequential(*layers)

    def get_layers(self, in_c, out_c, k=4, s=2, p=1, norm=True):
        layers = [spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=k,
                                          stride=s, padding=p))]
        if norm:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        labels = self.real_label if target_is_real else self.fake_label
        return labels.expand_as(preds)

    def forward(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss
    

class PerceptualLoss(nn.Module):
    def __init__(self, feature_layers=[0, 5, 10, 19, 28], weights=[1.0]*5):
        super(PerceptualLoss, self).__init__()
        vgg_weights = VGG16_Weights.DEFAULT
        self.vgg = vgg16(weights=vgg_weights).features[:max(feature_layers)+1].to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.feature_layers = feature_layers
        self.weights = weights

    def forward(self, pred, target):
        # Since the predicted and target images are ab channels, we need to create 3-channel images
        # We'll concatenate the L channel with the ab channels to form Lab images, then convert to RGB
        # For perceptual loss, we need RGB images with 3 channels

        # Reconstruct Lab images
        L = torch.zeros_like(pred[:, :1, :, :]).to(device)  # Dummy L channel
        pred_lab = torch.cat([L, pred], dim=1)
        target_lab = torch.cat([L, target], dim=1)

        # Convert Lab to RGB
        pred_rgb = lab_to_rgb(L, pred)
        target_rgb = lab_to_rgb(L, target)

        # Convert to tensors
        pred_rgb = torch.from_numpy(pred_rgb).permute(0, 3, 1, 2).to(device).float()
        target_rgb = torch.from_numpy(target_rgb).permute(0, 3, 1, 2).to(device).float()

        # Normalize RGB images to [-1, 1]
        pred_rgb = (pred_rgb / 0.5) - 1.0
        target_rgb = (target_rgb / 0.5) - 1.0

        loss = 0.0
        x = pred_rgb
        y = target_rgb
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in self.feature_layers:
                loss += self.weights[self.feature_layers.index(i)] * nn.functional.l1_loss(x, y)
        return loss
    

