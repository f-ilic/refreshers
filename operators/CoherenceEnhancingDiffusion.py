import torch
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize
import numpy as np
import torch.nn.functional as F

def get_Kx(width, height):
    # build gradient x operator, sparse matrix. use Kx.matmul(Image), or Kx.to_dense() to view
    # all values above the main diagonal for dx
    MN = width*height

    cidx = torch.Tensor(range(MN-1)).int() + 1
    ridx = torch.Tensor(range(MN-1)).int()
    i1 = torch.stack((ridx, cidx))
    v1 = torch.Tensor([1/2]).tile(MN-1)

    # all values below the main diagonal for dx
    cidx = torch.Tensor(range(MN-1)).int()
    ridx = torch.Tensor(range(MN-1)).int() + 1
    i2 = torch.stack((ridx, cidx))
    v2 = torch.Tensor([-1/2]).tile(MN-1)

    indices = torch.cat((i1, i2), dim=1)
    values = torch.cat((v1, v2))

    Kx = torch.sparse_coo_tensor(indices, values, size=(MN,MN))
    return Kx

def get_Ky(width, height):
    # build gradient y operator, sparse matrix. use Ky.matmul(Image), or Ky.to_dense() to view
    # all values above the main diagonal for dy
    MN = width*height

    cidx = torch.Tensor(range(MN-1)).int() + width
    ridx = torch.Tensor(range(MN-1)).int()

    cidx = cidx[:-width]
    ridx = ridx[:-width]

    i1 = torch.stack((ridx, cidx))
    v1 = torch.Tensor([1/2]).tile(MN-width-1)

    # all values below the main diagonal for y
    cidx = torch.Tensor(range(MN-1)).int()
    ridx = torch.Tensor(range(MN-1)).int() + width

    cidx = cidx[:-width]
    ridx = ridx[:-width]

    i2 = torch.stack((ridx, cidx))
    v2 = torch.Tensor([-1/2]).tile(MN-width-1)

    indices = torch.cat((i1, i2), dim=1)
    values = torch.cat((v1, v2))

    Ky = torch.sparse_coo_tensor(indices, values, size=(MN,MN))
    return Ky

def test_grad_operator():
    u = data.camera()
    width, height = (300, 300)
    u = resize(u, (width, height))
    u = torch.from_numpy(u).float()
    u = torch.reshape(u, (width * height, 1))

    # ----------- Gradient X direction -----------
    # once with convolution, once with bigass op matrix
    Kx = get_Kx(width, height)

    cx = torch.Tensor([[ 0, 0, 0],
                       [-1/2, 0, 1/2],
                       [ 0, 0, 0]]).unsqueeze(0).unsqueeze(0)

    conv_x_u = F.conv2d(u.reshape(1,1,width, height), cx, bias=None)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(conv_x_u.squeeze(), cmap='gray')
    ax[0].set_title("conv(u, [-1/2, 0, 1/2])")

    Kxu = Kx.matmul(u).reshape((height, width))
    ax[1].imshow(Kxu, cmap='gray')
    ax[1].set_title("Kx * u")

    # ----------- Gradient Y direction -----------
    # once with convolution, once with bigass op matrix
    Ky = get_Ky(width, height)

    cy = torch.Tensor([[0, -1/2, 0],
                       [0, 0, 0],
                       [0, 1/2, 0]]).unsqueeze(0).unsqueeze(0)

    conv_y_u = F.conv2d(u.reshape(1, 1, width, height), cy, bias=None)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(conv_y_u.squeeze(), cmap='gray')
    ax[0].set_title("conv(x, [-1/2, 0, 1/2].T)")

    ax[1].imshow(Ky.matmul(u).reshape((height, width)), cmap='gray')
    ax[1].set_title("Ky * u")

    plt.show()

if __name__ == "__main__":
    print("Coherence Enhancing Diffusion with parameters ...")
    test_grad_operator()