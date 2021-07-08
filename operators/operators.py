import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize
import torch

# gradient in X direction operator
A = data.camera()
A = resize(A, (100, 100))
m, n = A.shape


indices = [] # todo
values = [] # todo

# torch.sparse_coo_tensor()

Idx_reference = np.gradient(A, axis=1)
Idy_reference = np.gradient(A, axis=0)

A = torch.from_numpy(A).float().cuda()

# only works reliably for square images
Kx = torch.zeros((m*n+2,m*n+2))
for col in range(m*n):
    for row in range(m*n):
        if col==row:
            Kx[row, col-1] = -1/2
            Kx[row, col+1] = 1/2
Kx = Kx[1:-1, 1:-1]
Kx = Kx.cuda()

Ky = torch.zeros((m*n,m*n))
for row in range(n, m*n-n):
    Ky[row, row-n] = -1/2
    Ky[row, row+n] = 1/2
Ky=Ky.cuda()

vectorized_image = A.reshape((m*n, 1)).cuda()

fig, ax = plt.subplots(2, 3)

ax[0][0].set_title("input image")
ax[0][0].imshow(A.cpu(), cmap='gray')

ax[0][1].imshow(Idx_reference)
ax[0][1].set_title("Idx_reference")

ax[0][2].imshow(Idy_reference)
ax[0][2].set_title("Idy_reference")

ax[1][1].imshow((Kx@vectorized_image).float().reshape((m,n)).cpu())
ax[1][1].set_title("Idx w/ operator")

ax[1][2].imshow((Ky@vectorized_image).float().reshape((m,n)).cpu())
ax[1][2].set_title("Idy w/ operator")

plt.show()