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
Ky = get_Ky(N,M).todense()
K = np.vstack((Kx, Ky))
# K = Kx

# image_path = 'water-castle.png'
image_path = 'lenna.png'

pil_img = Image.open(image_path)
width, height = pil_img.size

x = (PILToTensor()(pil_img.resize((N,M)).convert('L')).squeeze().float() / 255).squeeze().numpy().flatten()


# x = np.random.randn(N*M)
y = np.random.randn(2*N*M, 1)

lhs = (K @ x).dot(y).item()
rhs = x.dot(K.T @ y).item()
print(f'{lhs=:.2f}, {rhs=:.2f}')

for name, item in zip(['x', 'y', 'K', 'K.T'],[x,y,K,K.T]):
    print(f'{name}\t: {item.shape}')


fig, ax = plt.subplots(1,4)
ax[0].imshow(x.reshape(N,M), cmap='gray')
grad_img_matrix = torch.Tensor((K @ x)).reshape((2,N,M))
tmp1 = (K @ x).reshape((2*N,M))#[10:-10, 10:-10]
ax[1].imshow(tmp1, cmap='gray')

print(f'##### ----------------------------------------')
print(f'Adjointness test for convolution')
# K =  torch.Tensor([[0, 0, 0],
#                    [-1 , 0, 1 ],
#                    [0, 0, 0]]).unsqueeze(0).unsqueeze(0).double()
Kx =  torch.Tensor([[0, 0, 0],
                   [0, -1, 1 ],
                   [0, 0, 0]]).double()

Ky =  torch.Tensor([[0, 0, 0],
                   [0, -1, 0 ],
                   [0, 1, 0]]).double()

K = torch.stack([Kx, Ky]).unsqueeze(1)
x = torch.from_numpy(x).reshape((1,1,N,M)).double()
y = torch.from_numpy(y).reshape((1, 2,N,M)).double()

padded = torch.nn.functional.pad(x, (1, 1, 1, 1), "constant", 0) # zero pad
grad_img = F.conv2d(padded, K, padding=0)
grad_x = grad_img.flatten() # Kx

div_x = F.conv_transpose2d(y, K, padding=1).flatten() # K'y

lhs = grad_x.dot(y.flatten()).sum().item()
rhs = x.flatten().dot(div_x).sum().item()


# print(f'all close: {torch.allclose(x.flatten().dot(div_x), grad_x.dot(y.flatten()))}')

tmp2 = grad_x.reshape(2*N,M)#[10:-10, 10:-10]
ax[2].imshow(tmp2, cmap='gray')

ax[3].imshow((tmp1-tmp2.numpy())*100, cmap='gray')
plt.figure()
plt.imshow(grad_img_matrix.sum(dim=0).squeeze(0), cmap='gray')
plt.figure()
plt.imshow(grad_img.sum(dim=1).squeeze(0), cmap='gray')
plt.show()

print(f'{lhs=:.2f}, {rhs=:.2f}')