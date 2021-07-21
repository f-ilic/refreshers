import torch
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize
import numpy as np
import torch.nn.functional as F
import math
from kornia.filters import gaussian_blur2d
from scipy.sparse import coo_matrix, identity, vstack, hstack, eye
from scipy.sparse.linalg import spsolve, lsqr
from PIL import Image

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

def test_div_operator():
    # TODO
    # transposity test  <Kx, y> = <x, K.T y>
    pass


def test_grad_operator():
    # transposity test  <Kx, y> = <x, K.T y>
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


def blur(x, sigma, kernelsz, width, height): # x: numpy array HW
    out = gaussian_blur2d(torch.Tensor(x).float().reshape((1, 1, width, height)), (kernelsz, kernelsz),
                         (sigma, sigma)).reshape((width*height, 1)).numpy()
    return out # numpy array [WH,1]


if __name__ == "__main__":
    # du/dt = div(D ∇u)
    # D... structure tensor
    # https://link.springer.com/content/pdf/10.1023/A:1008009714131.pdf

    # u = data.camera()
    u = np.asarray(Image.open('lenna.png').convert('L'))
    width, height = (110, 110)
    WH = width * height

    u = resize(u, (width, height))
    u = u + np.random.randn(u.shape[0], u.shape[1]) * 0.001

    u = torch.from_numpy(u).float()

    Kx = get_Kx(width, height)
    Ky = get_Ky(width, height)

    kernel1_sz, sigma1 = (3, 0.5)
    kernel2_sz, sigma2 = (7, 1)
    alpha = 0.0005
    gamma = 0.0001
    tau = 4

    u_vec = u.reshape((WH,1))
    usmooth = blur(u,sigma1,kernel1_sz, width, height)

    for t in range(0, 100):
        print(f'{t=}')

        dx = Kx @ usmooth
        dy = Ky @ usmooth

        # ------- compute the entries in the structure tensor
        # smooth the gradients first so we dont get shit
        dx = blur(dx, sigma2, kernel2_sz, width, height)
        dy = blur(dy, sigma2, kernel2_sz, width, height)

        dx2 = blur(dx**2, sigma2, kernel2_sz, width, height)
        dy2 = blur(dy**2, sigma2, kernel2_sz, width, height)
        dxy = blur(dx * dy , sigma2, kernel2_sz, width, height)

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

        # u_vec = spsolve(ALL, u_vec)
        u_vec = lsqr(ALL, u_vec)[0]
        # plt.imshow(u_vec.reshape(width, height), cmap='gray')
        plt.imsave(f'{t}.jpg',u_vec.reshape(width, height), cmap='gray')


