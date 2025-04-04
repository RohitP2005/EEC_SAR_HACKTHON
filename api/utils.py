import torch
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
import numpy as np


# AverageMeter for tracking losses
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count

# Function to convert Lab to RGB
def lab_to_rgb(L, ab):
    L = (L + 1.) * 50.  # Denormalize L channel from [-1, 1] to [0, 100]
    ab = ab * 128.  # Denormalize ab channels from [-1, 1] to [-110, 110]
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().detach().numpy()  # Shape: [batch_size, H, W, 3]
    rgb_imgs = []
    for img in Lab:
        img_rgb = np.clip(lab2rgb(img.astype('float64')), 0, 1)  # Convert LAB to RGB and clip values
        rgb_imgs.append((img_rgb * 255).astype(np.uint8))  # Convert to 8-bit format
    return np.stack(rgb_imgs, axis=0)

# Visualization function
def visualize(model, data, save=False, epoch=0):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        if i == 0:
            ax.set_title('Input L (SAR)')
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        if i == 0:
            ax.set_title('Generated RGB')
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
        if i == 0:
            ax.set_title('Ground Truth RGB')
    plt.tight_layout()
    if save:
        save_path = f"outputs/images/colorization_epoch_{epoch}.png"
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    plt.show()