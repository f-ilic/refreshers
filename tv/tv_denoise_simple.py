from torchvision.transforms import PILToTensor
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



class TVDenoise(torch.nn.Module):
    def __init__(self, noisy_image, reference_image, lmbda):
        super(TVDenoise, self).__init__()
        self.lmbda = lmbda
        self.reference_image = reference_image

        self.denoised_image = torch.clone(noisy_image)
        self.denoised_image.requires_grad = True

    def R(self, x):
        x_diff = x - torch.roll(x, -1, dims=1)
        y_diff = x - torch.roll(x, -1, dims=0)
        norm = torch.sum(torch.pow(x_diff, 2) + torch.pow(y_diff, 2))
        return norm

    def forward(self):
        return self.lmbda * torch.nn.MSELoss()(self.denoised_image, self.reference_image) + (1-self.lmbda) * self.R(self.denoised_image)

    def get_denoised_image(self):
        return self.denoised_image


image_path = 'lenna.png'
reference_image = PILToTensor()(Image.open(image_path).convert('L').resize((128, 128))).squeeze().float() / 255

noise = torch.randn_like(reference_image) * 0.5
noisy_image = reference_image + noise
noisy_image = np.clip(noisy_image, 0.0, 1.0)
noisy_image = torch.FloatTensor(noisy_image)


tv_denoiser = TVDenoise(noisy_image, reference_image, lmbda=0.998)
optimizer = torch.optim.SGD([tv_denoiser.denoised_image], lr=0.1, momentum=0.9)

num_iters = 500
for i in range(num_iters):
    optimizer.zero_grad()
    loss = tv_denoiser()
    if i % 25 == 0:
        print(f"Loss in iteration {i}/{num_iters}: {loss.item():.3f}")
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