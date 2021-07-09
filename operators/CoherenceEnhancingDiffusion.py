import torch
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize
import numpy as np
import torch.nn.functional as F
import math
from kornia.filters import gaussian_blur2d


def get_Kx(width, height):
    # build gradient x operator, sparse matrix. use Kx.matmul(Image), or Kx.to_dense() to view
    # all values above the main diagonal for dx
    MN = width * height

    cidx = torch.Tensor(range(MN - 1)).int() + 1
    ridx = torch.Tensor(range(MN - 1)).int()
    i1 = torch.stack((ridx, cidx))
    v1 = torch.Tensor([1 / 2]).tile(MN - 1)

    # all values below the main diagonal for dx
    cidx = torch.Tensor(range(MN - 1)).int()
    ridx = torch.Tensor(range(MN - 1)).int() + 1
    i2 = torch.stack((ridx, cidx))
    v2 = torch.Tensor([-1 / 2]).tile(MN - 1)

    indices = torch.cat((i1, i2), dim=1)
    values = torch.cat((v1, v2))

    Kx = torch.sparse_coo_tensor(indices, values, size=(MN, MN))
    return Kx


def get_Ky(width, height):
    # build gradient y operator, sparse matrix. use Ky.matmul(Image), or Ky.to_dense() to view
    # all values above the main diagonal for dy
    MN = width * height

    cidx = torch.Tensor(range(MN - 1)).int() + width
    ridx = torch.Tensor(range(MN - 1)).int()

    cidx = cidx[:-width]
    ridx = ridx[:-width]

    i1 = torch.stack((ridx, cidx))
    v1 = torch.Tensor([1 / 2]).tile(MN - width - 1)

    # all values below the main diagonal for y
    cidx = torch.Tensor(range(MN - 1)).int()
    ridx = torch.Tensor(range(MN - 1)).int() + width

    cidx = cidx[:-width]
    ridx = ridx[:-width]

    i2 = torch.stack((ridx, cidx))
    v2 = torch.Tensor([-1 / 2]).tile(MN - width - 1)

    indices = torch.cat((i1, i2), dim=1)
    values = torch.cat((v1, v2))

    Ky = torch.sparse_coo_tensor(indices, values, size=(MN, MN))
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

    cx = torch.Tensor([[0, 0, 0],
                       [-1 / 2, 0, 1 / 2],
                       [0, 0, 0]]).unsqueeze(0).unsqueeze(0)

    conv_x_u = F.conv2d(u.reshape(1, 1, width, height), cx, bias=None)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(conv_x_u.squeeze(), cmap='gray')
    ax[0].set_title("conv(u, [-1/2, 0, 1/2])")

    Kxu = Kx.matmul(u).reshape((height, width))
    ax[1].imshow(Kxu, cmap='gray')
    ax[1].set_title("Kx * u")

    # ----------- Gradient Y direction -----------
    # once with convolution, once with bigass op matrix
    Ky = get_Ky(width, height)

    cy = torch.Tensor([[0, -1 / 2, 0],
                       [0, 0, 0],
                       [0, 1 / 2, 0]]).unsqueeze(0).unsqueeze(0)

    conv_y_u = F.conv2d(u.reshape(1, 1, width, height), cy, bias=None)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(conv_y_u.squeeze(), cmap='gray')
    ax[0].set_title("conv(x, [-1/2, 0, 1/2].T)")

    ax[1].imshow(Ky.matmul(u).reshape((height, width)), cmap='gray')
    ax[1].set_title("Ky * u")

    plt.show()


if __name__ == "__main__":
    print("Coherence Enhancing Diffusion with parameters ...")
    # du/dt = div(D ∇u)
    # D... structure tensor
    # https://link.springer.com/content/pdf/10.1023/A:1008009714131.pdf

    u = data.camera()
    width, height = (80, 80)
    WH = width * height

    u = resize(u, (width, height))
    u = torch.from_numpy(u).float()

    Kx = get_Kx(width, height)
    Ky = get_Ky(width, height)

    kernel1_sz, sigma1 = (7, 0.8)
    kernel2_sz, sigma2 = (3, 0.5)
    alpha = 0.0005
    gamma = 0.0001
    tau = 10

    u = gaussian_blur2d(u.unsqueeze(0).unsqueeze(0),
                        (kernel1_sz, kernel1_sz),
                        (sigma1, sigma1))

    u_vec = u.reshape((width * height, 1))
    print(f'{u_vec.shape}')
    plt.ioff()
    for t in range(0, 100, 10):

        dx = Kx.matmul(u_vec)
        dy = Ky.matmul(u_vec)

        # ------- compute the entries in the structure tensor
        # smooth the gradients first so we dont get shit
        dx = gaussian_blur2d(dx.reshape((1, 1, width, height)), (kernel2_sz, kernel2_sz), (sigma2, sigma2)).reshape((WH, 1))
        dy = gaussian_blur2d(dy.reshape((1, 1, width, height)), (kernel2_sz, kernel2_sz), (sigma2, sigma2)).reshape((WH, 1))

        dx2 = dx * dx
        dy2 = dy * dy
        dxy = dx * dy

        trace = dx2 + dy2
        det = (dx2 * dy2) - (dxy * dxy)

        # eigenvalues
        tmp1 = trace * 1 / 2
        tmp2 = ((trace ** 2) / 4 - det) ** (1 / 2)

        lmbda1 = tmp1 + tmp2
        lmbda2 = tmp1 - tmp2

        # eigenvectors

        v1 = torch.cat((torch.ones(WH, 1), (lmbda1 - dx2) / dxy), dim=1)
        v2 = torch.cat((torch.ones(WH, 1), (lmbda2 - dx2) / dxy), dim=1)

        # convert eigenvectors to unit-vectors
        v1_unit = v1 / torch.norm(v1, dim=1)[:, None]
        v2_unit = v2 / torch.norm(v2, dim=1)[:, None]

        l1 = alpha
        l2 = alpha + (1 - alpha) * (1 - abs(torch.exp(-(tmp1 - tmp2) ** 2) / (2 * gamma ** 2)))

        a = v1_unit[:, 0] * v1_unit[:, 0] * l1 + v2_unit[:, 0] * v2_unit[:, 0] * l2.squeeze()
        b = v1_unit[:, 0] * v1_unit[:, 1] * l1 + v2_unit[:, 0] * v2_unit[:, 1] * l2.squeeze()
        c = b
        d = v1_unit[:, 1] * v1_unit[:, 1] * l1 + v2_unit[:, 1] * v2_unit[:, 1] * l2.squeeze()

        # Assemble sparse structure tensor
        diags = torch.Tensor(range(WH)).int()
        indices = diags.repeat(2,1)

        a_diag = torch.sparse_coo_tensor(indices, a, size=(WH, WH))
        b_diag = torch.sparse_coo_tensor(indices, b, size=(WH, WH))
        c_diag = torch.sparse_coo_tensor(indices, c, size=(WH, WH))
        d_diag = torch.sparse_coo_tensor(indices, d, size=(WH, WH))


        D = torch.vstack((torch.hstack((a_diag,b_diag)),
                          torch.hstack((c_diag,d_diag))))
        K = torch.cat((Kx,Ky), dim=0)
        identity = np.identity(WH)
        ALL = identity + tau * (K.to_dense().T @ D.to_dense() @ K.to_dense()).numpy()
        u_vec = torch.from_numpy(np.linalg.lstsq(ALL, u_vec)[0]).float()
        print(f'{u_vec.shape}')
        plt.imsave(f'{t}.jpg',u.reshape(width, height), cmap='gray')


