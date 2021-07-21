import matplotlib.pyplot
import torch.nn
import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize
import numpy as np
import torch.nn.functional as F
import math
import scipy
from kornia.filters import gaussian_blur2d
from scipy.sparse import coo_matrix, identity, vstack, hstack, eye
from scipy.sparse.linalg import spsolve, lsqr
from PIL import Image
import torch.nn.modules

def get_Kx(N, M):
    a = np.append(np.ones(N-1), 0)
    dx = scipy.sparse.diags([np.tile(-a, M),np.tile(a, M)], [0, 1], (M*N, M*N))
    return dx.tocsr()

def get_Ky(N,M):
    a = np.append(np.tile(np.ones(N), M-1), np.zeros(N))
    dy = scipy.sparse.diags([-a, a], [0, N], (M*N, M*N))
    return dy.tocsr()


N=10
M=10

print(f'##### ----------------------------------------')
print(f'Adjointness test for operator')
Kx = get_Kx(N,M).todense()
# Ky = get_Ky(N,M).todense()
# K = np.vstack((Kx, Ky))
K = Kx
for i in range(10):
    x = np.random.randn(N*M)
    y = np.random.randn(N*M, 1)

    lhs = (K @ x).dot(y).item()
    rhs = x.dot(K.T @ y).item()
    print(f'{i}: {lhs=:.2f}, {rhs=:.2f}')

for name, item in zip(['x', 'y', 'K', 'K.T'],[x,y,K,K.T]):
    print(f'{name}\t: {item.shape}')

print(f'##### ----------------------------------------')
print(f'Adjointness test for convolution')
K =  torch.Tensor([[0, 0, 0],
                   [-1 / 2, 0, 1 / 2],
                   [0, 0, 0]]).unsqueeze(0).unsqueeze(0).double()

x = torch.from_numpy(x).reshape((N,M)).unsqueeze(0).unsqueeze(0).double()
y = torch.from_numpy(y).reshape((N,M)).unsqueeze(0).unsqueeze(0).double()

grad_x = F.conv2d(x, K, padding=1).flatten() # Kx

div_x = F.conv_transpose2d(y, K, padding=1).flatten() # K'y

lhs = grad_x.dot(y.flatten()).sum().item()
rhs = x.flatten().dot(div_x).sum().item()

print(f'{lhs=:.2f}, {rhs=:.2f}')