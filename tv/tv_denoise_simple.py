from torchvision.transforms import PILToTensor
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def R(x):
    x_diff = x - torch.roll(x, -1, dims=1)
    y_diff = x - torch.roll(x, -1, dims=0)
    norm = x_diff.abs().sum() + y_diff.abs().sum()
    return norm

class TVDenoise(torch.nn.Module):
    def __init__(self, noisy_image, tau):
        super(TVDenoise, self).__init__()
        self.tau = tau
        self.reference_image = torch.clone(noisy_image)
        self.reference_image.requires_grad = False

        self.denoised_image = torch.clone(noisy_image)
        self.denoised_image.requires_grad = True

    def forward(self):
        return torch.nn.MSELoss()(self.denoised_image, self.reference_image) + self.tau * R(self.denoised_image)

    def get_denoised_image(self):
        return self.denoised_image


image_path = 'face.jpg'
# image_path = 'water-castle.png'
image_path = 'lenna.png'
# image_path = 'markus.png'

pil_img = Image.open(image_path)
width, height = pil_img.size

reference_image = PILToTensor()(pil_img.resize((width//2, height//2)).convert('L')).squeeze().float() / 255

# Denoising
noise = torch.randn_like(reference_image) * 0.1
noisy_image = reference_image + noise
noisy_image = np.clip(noisy_image, 0.0, 1.0)
noisy_image = torch.FloatTensor(noisy_image)

# Inpainting
# mask = torch.FloatTensor(height//2, width//2).uniform_() > 0.5
# noisy_image = reference_image * mask

tv_denoiser = TVDenoise(noisy_image, tau=0.0095)
optimizer = torch.optim.SGD([tv_denoiser.denoised_image], lr=0.1)

num_iters = 500
for i in range(num_iters):
    optimizer.zero_grad()
    loss = tv_denoiser()
    if i % 100 == 0:
        print(f"Loss in iteration {i}/{num_iters}: {loss.item():.3f}")
        # denoised_image = torch.clone(tv_denoiser.get_denoised_image())
        # plt.imshow(denoised_image.detach().numpy(), cmap='gray')
        # plt.axis('off')
        # plt.show()

    loss.backward()
    optimizer.step()

denoised_image = tv_denoiser.get_denoised_image()


fig, axs = plt.subplots(1, 3, figsize=(16, 10))
axs = axs.ravel()

axs[0].axis('off')
axs[0].set_title('Reference')
axs[0].imshow(reference_image, cmap='gray')

axs[1].axis('off')
axs[1].set_title('Noisy image')
axs[1].imshow(noisy_image, cmap='gray')

axs[2].axis('off')
axs[2].set_title('Denoised image')
axs[2].imshow(denoised_image.detach().numpy(), cmap='gray')

plt.show()