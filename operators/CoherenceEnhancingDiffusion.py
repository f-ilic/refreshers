import torch
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize

def get_Kx_shitty(width, height):
    mn = width*height
    Kx = torch.zeros((mn + 2, mn + 2))
    for col in range(mn):
        for row in range(mn):
            if col == row:
                Kx[row, col - 1] = -1 / 2
                Kx[row, col + 1] = 1 / 2
    Kx = Kx[1:-1, 1:-1]
    return Kx

def get_Ky_shitty(width, height):
    mn = width*height
    Ky = torch.zeros((mn,mn))
    for row in range(height, mn - height):
        Ky[row, row - height] = -1 / 2
        Ky[row, row + height] = 1 / 2
    return Ky

def get_Kx(width, height):
    # build gradient x operator, sparse matrix. use Kx.matmul(Image), or Kx.to_dense() to view
    # all values above the main diagonal for dx
    cidx = torch.Tensor(range(width-1)).int() + 1
    ridx = torch.Tensor(range(height-1)).int()
    i1 = torch.stack((ridx, cidx))
    v1 = torch.Tensor([1/2]).tile(width-1)

    # all values below the main diagonal for dx
    cidx = torch.Tensor(range(width - 1)).int()
    ridx = torch.Tensor(range(height - 1)).int() + 1
    i2 = torch.stack((ridx, cidx))
    v2 = torch.Tensor([-1/2]).tile(width - 1)

    indices = torch.cat((i1, i2), dim=1)
    values = torch.cat((v1, v2))

    Kx = torch.sparse_coo_tensor(indices, values, size=(height, width))
    return Kx

def get_Ky(width, height):
    pass

if __name__ == "__main__":
    print("Coherence Enhancing Diffusion with parameters ...")
    u = data.camera()
    width, height = (300, 300)
    u = resize(u, (width, height))
    u = torch.from_numpy(u).float()

    Kx = get_Kx(width, height)
    plt.imshow(Kx.matmul(u), cmap='gray')
    plt.show()