import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import data
from skimage import io
from scipy.ndimage import convolve
from skimage.color import rgb2gray
from skimage.transform import rescale
from matplotlib.widgets import Button, Slider


rc = {"axes.spines.left": False,
      "axes.spines.right": False,
      "axes.spines.bottom": False,
      "axes.spines.top": False,
      "xtick.bottom": False,
      "xtick.labelbottom": False,
      "ytick.labelleft": False,
      "ytick.left": False}
plt.rcParams.update(rc)

operator_dy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
operator_dx = operator_dy.transpose()



def image_deriv_demo():
    fig, ax = plt.subplots(3, 3)

    def circle_img(h, w, center=None, radius=None):
        if center is None:  # use the middle of the image
            center = (int(w / 2), int(h / 2))
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= radius
        return mask

    # original = data.astronaut()
    # I = rgb2gray(original)
    I = np.clip(gaussian_filter(circle_img(25, 25) * 1., 1), 0, 1)
    I = np.pad(I, 5)  # just add a bit of border

    Idy = convolve(I, operator_dy)
    Idx = convolve(I, operator_dx)

    ax[0][0].set_title('Input Image')
    ax[0][0].imshow(I)

    ax[1][0].set_title('dx operator')
    ax[1][0].imshow(operator_dx, cmap='bwr')

    ax[1][1].set_title('dy operator')
    ax[1][1].imshow(operator_dy, cmap='bwr')

    ax[2][0].set_title('I dx')
    ax[2][0].imshow(Idx, cmap='bwr')

    ax[2][1].set_title('I dy')
    ax[2][1].imshow(Idy, cmap='bwr')
    plt.show()





def find_path(pm, startidx):
    h, w = pm.shape
    path = [startidx]

    prev_min = startidx

    for row in range(1, h):
        prev_min += int(pm[row, prev_min])
        path.append(prev_min)
    return path

def path_matrix(c):
    x = np.zeros_like(c)
    h, w = c.shape
    for row in range(0, h-1):
        for col in range(0, w):
            if col - 1 < 0:
                dir = np.argmin(c[row + 1, col:col+2])
                x[row,col] = (0,1)[dir]

            elif col + 1 > w:
                dir = np.argmin(c[row + 1, col-1:col + 1])
                x[row, col] = (-1, 0)[dir]

            else:
                dir = np.argmin(c[row + 1, col - 1:col + 2])
                x[row, col] = (-1, 0, 1)[dir]
    return x


def cost(x):
    c = np.zeros_like(x)
    h, w = c.shape

    c[-1, :] = x[-1, :]
    for row in reversed(range(0, h - 1)):
        for col in range(0, w):
            lborder = rborder = 0
            if col - 1 < 0: lborder = 1
            if col + 1 > w: rborder = 1

            c[row][col] = np.min(c[row+1,col-1+lborder:col+2-rborder]) + x[row][col]
    return c




def main():
    fig, ax = plt.subplots(1,3)
    ax = ax.flatten()
    original = io.imread('dali.jpg')
    I = rescale(rgb2gray(original), 0.3)

    data = {'I': [],
            'Idxdy_magnitude': [],
            'Cost': [],
            'Seam_img': []
            }

    for i in range(200):
        print(f"In iteration {i}")
        Idy = convolve(I, operator_dy)
        Idx = convolve(I, operator_dx)

        Idxdy_magnitude = np.abs(Idy) + np.abs(Idx)

        c = cost(Idxdy_magnitude)

        dirs = path_matrix(c)
        cols = range(0, c.shape[0])

        startidx = c[0,:].argmin()
        seam = find_path(dirs, startidx)

        seam_img = np.zeros_like(I)
        seam_img[cols, seam] = 1

        I = remove_seam(I, c, seam)

        data['I'].append(I)
        data['Idxdy_magnitude'].append(Idxdy_magnitude)
        data['Cost'].append(c)
        data['Seam_img'].append(seam_img)

    # plot the stuff

    ax_p = plt.axes([0.15, 0.02, 0.30, 0.03])
    slider_p = Slider(ax_p, 'Horizontal Compression', 0, len(data['I'])-1, 0, valstep=1)


    def update_slider(self):
        slice = int(slider_p.val)
        for a in ax:
            a.clear()
        ax[0].set_title('Input Image')
        ax[0].imshow(data['I'][slice], cmap='gray')
        ax[1].set_title('|I dx| + |I dy|')
        ax[1].imshow(data['Idxdy_magnitude'][slice])
        ax[2].set_title('Cost')
        ax[2].imshow(data['Cost'][slice])
        # ax[3].imshow(seam_img)

    slider_p.on_changed(update_slider)
    plt.ioff()
    plt.show()


def remove_seam(I, c, seam):
    h, w = c.shape
    mask = np.ones((h, w), dtype=bool)
    mask[range(h), seam] = False
    I = I[mask].reshape(h, w - 1)
    return I



if __name__ == "__main__":
    main()