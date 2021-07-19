import torch
from torch.nn import Softmax
import matplotlib.pyplot as plt

def create_1D_Gaussian_kernel(standard_deviation: float) -> torch.FloatTensor:
    kernel = torch.FloatTensor()
    torch.pi = torch.acos(torch.zeros(1)).item() *2 # define pi from torch
    kernel_size_k = 4 * standard_deviation + 1
    kernel = torch.arange(kernel_size_k)
    mean_mu = kernel_size_k / 2
    variance = standard_deviation ** 2
    normalization_Z = 1 / (torch.sqrt(torch.tensor(2 * torch.pi)) * standard_deviation)
    kernel = normalization_Z * torch.exp(-((kernel - mean_mu) ** 2) / (2 * variance))
    return kernel
x = create_1D_Gaussian_kernel(6)

temps = torch.arange(0.1,2,0.3)
fig, ax = plt.subplots(1, len(temps), sharey=True)

for i, T in enumerate(temps):
    ax[i].set_title(f'T: {T:.1f}')
    ax[i].plot(Softmax(dim=0)(x/T))
    ax[i].grid('on')

plt.show()