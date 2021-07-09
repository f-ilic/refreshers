import torch
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize
import numpy as np
import torch.nn.functional as F
import math
from kornia.filters import gaussian_blur2d
from scipy.sparse import coo_matrix, identity, vstack, hstack, eye
from PIL import Image

# def get_Kx(width, height):
#     # build gradient x operator, sparse matrix. use Kx.matmul(Image), or Kx.to_dense() to view
#     # all values above the main diagonal for dx
#     MN = width * height
#
#     cidx = torch.Tensor(range(MN - 1)).int() + 1
#     ridx = torch.Tensor(range(MN - 1)).int()
#     i1 = torch.stack((ridx, cidx))
#     v1 = torch.Tensor([1 / 2]).tile(MN - 1)
#
#     # all values below the main diagonal for dx
#     cidx = torch.Tensor(range(MN - 1)).int()
#     ridx = torch.Tensor(range(MN - 1)).int() + 1
#     i2 = torch.stack((ridx, cidx))
#     v2 = torch.Tensor([-1 / 2]).tile(MN - 1)
#
#     indices = torch.cat((i1, i2), dim=1)
#     values = torch.cat((v1, v2))
#
#     Kx = torch.sparse_coo_tensor(indices, values, size=(MN, MN))
#     return Kx

def get_Kx(width, height):
    # build gradient x operator, sparse matrix. use Kx.matmul(Image), or Kx.to_dense() to view
    # all values above the main diagonal for dx
    MN = width * height

    cols1 = np.array(range(MN - 1)) + 1
    rows1 = np.array(range(MN - 1))
    values1 = np.tile(np.array([1 / 2]), MN-1)

    # all values below the main diagonal for dx
    cols2 = np.array(range(MN - 1))
    rows2 = np.array(range(MN - 1)) + 1
    values2 = np.tile(np.array([-1 / 2]), MN-1)


    values = np.hstack((values1, values2))
    cols = np.hstack((cols1, cols2))
    rows = np.hstack((rows1, rows2))

    Kx = coo_matrix((values, (cols, rows)), shape=(WH,WH))
    return Kx

def get_Ky(width, height):
    MN = width * height

    cols1 = np.array(range(MN - 1)) + width
    rows1 = np.array(range(MN - 1))
    values1 = np.tile(np.array([1 / 2]), MN-1- width)

    cols1 = cols1[:-width]
    rows1 = rows1[:-width]

    # all values below the main diagonal for dx
    cols2 = np.array(range(MN - 1))
    rows2 = np.array(range(MN - 1)) + width
    values2 = np.tile(np.array([-1 / 2]), MN-1- width)

    cols2 = cols2[:-width]
    rows2 = rows2[:-width]

    values = np.hstack((values1, values2))
    cols = np.hstack((cols1, cols2))
    rows = np.hstack((rows1, rows2))

    Ky = coo_matrix((values, (cols, rows)), shape=(WH,WH))
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


def slow_and_only_small_img_working():
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

    u_vec = u.reshape((width * height, 1)).numpy()

    print(f'{u_vec.shape}')
    plt.ioff()
    for t in range(0, 100, 10):
        dx = Kx.matmul(u_vec)
        dy = Ky.matmul(u_vec)

        # ------- compute the entries in the structure tensor
        # smooth the gradients first so we dont get shit
        dx = gaussian_blur2d(dx.reshape((1, 1, width, height)), (kernel2_sz, kernel2_sz), (sigma2, sigma2)).reshape(
            (WH, 1))
        dy = gaussian_blur2d(dy.reshape((1, 1, width, height)), (kernel2_sz, kernel2_sz), (sigma2, sigma2)).reshape(
            (WH, 1))

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
        indices = diags.repeat(2, 1)

        a_diag = torch.sparse_coo_tensor(indices, a, size=(WH, WH))
        b_diag = torch.sparse_coo_tensor(indices, b, size=(WH, WH))
        c_diag = torch.sparse_coo_tensor(indices, c, size=(WH, WH))
        d_diag = torch.sparse_coo_tensor(indices, d, size=(WH, WH))

        D = torch.vstack((torch.hstack((a_diag, b_diag)),
                          torch.hstack((c_diag, d_diag))))
        K = torch.cat((Kx, Ky), dim=0)
        identity = np.identity(WH)
        ALL = identity + tau * (K.to_dense().T @ D.to_dense() @ K.to_dense()).numpy()
        u_vec = torch.from_numpy(np.linalg.lstsq(ALL, u_vec)[0]).float()
        print(f'{u_vec.shape}')
        plt.imsave(f'{t}.jpg', u.reshape(width, height), cmap='gray')


if __name__ == "__main__":
    # du/dt = div(D ∇u)
    # D... structure tensor
    # https://link.springer.com/content/pdf/10.1023/A:1008009714131.pdf

    # u = data.camera()
    u = np.asarray(Image.open('gandalf.jpg').convert('L'))
    width, height = (140, 140)
    WH = width * height

    u = resize(u, (width, height))
    u = torch.from_numpy(u).float()

    Kx = get_Kx(width, height)
    Ky = get_Ky(width, height)

    kernel1_sz, sigma1 = (7, 0.5)
    kernel2_sz, sigma2 = (3, 0.3)
    alpha = 0.0005
    gamma = 0.0001
    tau = 10

    u = gaussian_blur2d(u.unsqueeze(0).unsqueeze(0),
                        (kernel1_sz, kernel1_sz),
                        (sigma1, sigma1))

    u_vec = u.reshape((width * height, 1)).numpy()
    print(f'{u_vec.shape}')
    plt.ioff()


    for t in range(0, 100):
        print(f'{t=}')

        dx = Kx @ u_vec
        dy = Ky @ u_vec

        # ------- compute the entries in the structure tensor
        # smooth the gradients first so we dont get shit
        dx = gaussian_blur2d(torch.Tensor(dx).float().reshape((1, 1, width, height)), (kernel2_sz, kernel2_sz), (sigma2, sigma2)).reshape((WH, 1)).numpy()
        dy = gaussian_blur2d(torch.Tensor(dy).float().reshape((1, 1, width, height)), (kernel2_sz, kernel2_sz), (sigma2, sigma2)).reshape((WH, 1)).numpy()

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

        v1 = np.hstack((np.ones(WH)[:,None], (lmbda1 - dx2) / dxy))
        v2 = np.hstack((np.ones(WH)[:,None], (lmbda2 - dx2) / dxy))

        # convert eigenvectors to unit-vectors
        v1_unit = v1 / np.linalg.norm(v1, axis=1)[:, None]
        v2_unit = v2 / np.linalg.norm(v2, axis=1)[:, None]

        l1 = alpha
        l2 = alpha + (1 - alpha) * (1 - abs(np.exp(-(tmp1 - tmp2) ** 2) / (2 * gamma ** 2)))

        a = v1_unit[:, 0] * v1_unit[:, 0] * l1 + v2_unit[:, 0] * v2_unit[:, 0] * l2.squeeze()
        b = v1_unit[:, 0] * v1_unit[:, 1] * l1 + v2_unit[:, 0] * v2_unit[:, 1] * l2.squeeze()
        c = b
        d = v1_unit[:, 1] * v1_unit[:, 1] * l1 + v2_unit[:, 1] * v2_unit[:, 1] * l2.squeeze()

        # Assemble sparse structure tensor
        diags = np.array(range(WH))
        a_diag = coo_matrix((a, (diags,diags)), shape=(WH, WH))
        b_diag = coo_matrix((b, (diags,diags)), shape=(WH, WH))
        c_diag = coo_matrix((c, (diags,diags)), shape=(WH, WH))
        d_diag = coo_matrix((d, (diags,diags)), shape=(WH, WH))

        D = vstack((hstack((a_diag,b_diag)), hstack((c_diag,d_diag))))

        K = vstack((Kx, Ky))
        id = identity(WH)

        ALL = id + tau * (K.transpose() @ D @ K)

        u_vec = np.linalg.lstsq(ALL.todense(), u_vec)
        plt.imsave(f'{t}.jpg', u.reshape(width, height), cmap='gray')


