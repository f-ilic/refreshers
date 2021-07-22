import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import scipy
from scipy.sparse import coo_matrix, identity, vstack, hstack, eye

import torch.nn.modules
from PIL import Image
from torchvision.transforms import PILToTensor

def get_Kx(N, M):
    a = np.append(np.ones(N-1), 0)
    dx = scipy.sparse.diags([np.tile(-a, M),np.tile(a, M)], [0, 1], (M*N, M*N))
    return dx.tocsr()

def get_Ky(N,M):
    a = np.append(np.tile(np.ones(N), M-1), np.zeros(N))
    dy = scipy.sparse.diags([-a, a], [0, N], (M*N, M*N))
    return dy.tocsr()


N=100
M=100

print(f'##### ----------------------------------------')
print(f'Adjointness test for operator')
Kx = get_Kx(N,M).todense()
# Ky = get_Ky(N,M).todense()
# K = np.vstack((Kx, Ky))
K = Kx

# image_path = 'water-castle.png'
image_path = 'lenna.png'

pil_img = Image.open(image_path)
width, height = pil_img.size

x = (PILToTensor()(pil_img.resize((N,M)).convert('L')).squeeze().float() / 255).squeeze().numpy().flatten()


# x = np.random.randn(N*M)
y = np.random.randn(N*M, 1)

lhs = (K @ x).dot(y).item()
rhs = x.dot(K.T @ y).item()
print(f'{lhs=:.2f}, {rhs=:.2f}')

for name, item in zip(['x', 'y', 'K', 'K.T'],[x,y,K,K.T]):
    print(f'{name}\t: {item.shape}')


fig, ax = plt.subplots(1,4)
ax[0].imshow(x.reshape(N,M), cmap='gray')
tmp1 = (K @ x).reshape((N,M))[10:-10, 10:-10]
ax[1].imshow(tmp1, cmap='gray')

print(f'##### ----------------------------------------')
print(f'Adjointness test for convolution')
# K =  torch.Tensor([[0, 0, 0],
#                    [-1 , 0, 1 ],
#                    [0, 0, 0]]).unsqueeze(0).unsqueeze(0).double()
K =  torch.Tensor([[0, 0, 0],
                   [0, -1, 1 ],
                   [0, 0, 0]]).unsqueeze(0).unsqueeze(0).double()


x = torch.from_numpy(x).reshape((N,M)).unsqueeze(0).unsqueeze(0).double()
y = torch.from_numpy(y).reshape((N,M)).unsqueeze(0).unsqueeze(0).double()

grad_x = F.conv2d(x, K, padding=1).flatten() # Kx

div_x = F.conv_transpose2d(y, K, padding=1).flatten() # K'y

lhs = grad_x.dot(y.flatten()).sum().item()
rhs = x.flatten().dot(div_x).sum().item()

tmp2 = grad_x.reshape(N,M)[10:-10, 10:-10]
ax[2].imshow(tmp2, cmap='gray')

ax[3].imshow((tmp1-tmp2.numpy())*100, cmap='gray')
plt.show()

print(f'{lhs=:.2f}, {rhs=:.2f}')