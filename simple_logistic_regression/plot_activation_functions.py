import torch
import numpy as np
import matplotlib.pyplot as plt
from generate_data import simple_dataset, donut_dataset, moon_dataset
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import seaborn as sns

# set dark theme
# sns.set_context("paper")

torch.manual_seed(1337)
activationFns = [
    torch.nn.Sigmoid(),
    torch.nn.Tanh(),
    torch.nn.GELU(),
    torch.nn.ReLU(),
    torch.nn.LeakyReLU(0.2),
]

fig, ax = plt.subplots(
    1, len(activationFns), sharex=True, sharey=True, figsize=(8, 8 / len(activationFns))
)
X = np.linspace(-2, 2, 1000)
for id, fun in enumerate(activationFns):
    ax[id].plot(X, fun(torch.Tensor(X)).numpy())
    ax[id].set_title(type(fun).__name__)

# set aspect ration 1:1
for a in ax:
    a.set_aspect(1)
    a.set_xlim(-2, 2)
    a.set_ylim(-2, 2)
    a.xaxis.grid(True, which="major", linestyle="-", linewidth=0.5)
    a.yaxis.grid(True, which="major", linestyle="-", linewidth=0.5)


fig.tight_layout()
plt.show()
