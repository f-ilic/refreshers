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
    pass

def get_Ky(width, height):
    pass

if __name__ == "__main__":
    print("Coherence Enhancing Diffusion with parameters ...")
    u = data.camera()
    width, height = (200, 200)
    u = resize(u, (width, height))
    u = torch.from_numpy(u).float()

    # Kx = get_Kx(width, height).cuda()
    # Ky = get_Ky(width, height).cuda()


    plt.imshow(u, cmap='gray')
    plt.show()