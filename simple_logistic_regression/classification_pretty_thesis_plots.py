import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from generate_data import simple_dataset, donut_dataset, moon_dataset, swiss_roll
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from matplotlib import cm, colors

torch.manual_seed(1337)
import matplotlib

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "30"
torch.pi = torch.acos(torch.zeros(1)).item() * 2

LAPTOP = True

if LAPTOP:
    # device = torch.device("mps")
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0")

print(f"alive and using {device}")


class SimpleLogisticRegression(torch.nn.Module):
    def __init__(self, activationFn=torch.nn.ReLU, num_classes=2):
        super(SimpleLogisticRegression, self).__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(3, num_classes)
        self.activation = activationFn()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        return self.softmax(x)


def plot_input_samples(activationFn, samples, labels):
    s = samples.cpu()
    plt.scatter(
        s[:, 0], s[:, 1], c=labels.cpu(), cmap="plasma", edgecolor="black", s=120
    )
    # set limits
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect("equal")
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.savefig(
        f"out/{activationFn.__name__}/input.pdf", aspect="equal", bbox_inches="tight"
    )

    return s


def plot_boundary_with_inputs(
    activationFn,
    labels,
    r2,
    r1,
    s,
    epoch,
    X,
    Y,
    grid_pred_logits,
    plot_cbar=True,
):
    fig, ax = plt.subplots(1, 1)
    ax.contourf(
        X,
        Y,
        grid_pred_logits[:, 1].reshape(r1, r2),
        cmap="plasma",
        alpha=0.7,
    )

    ax.contour(
        X,
        Y,
        grid_pred_logits[:, 1].reshape(r1, r2),
        colors="k",
        linewidths=3,
        linestyles="dashed",
    )
    # add colorbar
    norm = colors.Normalize(0, 1)
    cmap = cm.get_cmap("plasma")
    if plot_cbar:
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    ax.scatter(
        s[:, 0], s[:, 1], c=labels.cpu(), cmap="plasma", edgecolor="black", s=120
    )
    plt.gca().set_aspect("equal")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.gca().axes.xaxis.set_ticklabels([])

    plt.savefig(
        f"out/{activationFn.__name__}/out_epoch_{epoch}.pdf",
        aspect="equal",
        bbox_inches="tight",
    )


def plot_contours_only(
    activationFn, r2, r1, epoch, X, Y, grid_pred_logits, plot_cbar=True
):
    fig, ax = plt.subplots(1, 1)
    ax.contourf(
        X,
        Y,
        grid_pred_logits[:, 1].reshape(r1, r2),
        cmap="plasma",
        alpha=0.7,
    )

    ax.contour(
        X,
        Y,
        grid_pred_logits[:, 1].reshape(r1, r2),
        colors="k",
        linewidths=3,
        linestyles="dashed",
    )

    norm = colors.Normalize(0, 1)
    cmap = cm.get_cmap("plasma")
    if plot_cbar:
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

    plt.gca().set_aspect("equal")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.gca().axes.xaxis.set_ticklabels([])

    plt.savefig(
        f"out/{activationFn.__name__}/contour_epoch_{epoch}.pdf",
        aspect="equal",
        bbox_inches="tight",
    )


def main(activationFN):
    activationFn = activationFN
    lr = 0.05
    num_classes = 2  # or 3
    num_epochs = 400
    samples, labels = donut_dataset(num_classes=num_classes)
    r1 = r2 = 60

    # samples, labels = simple_dataset(num_samples=100, num_classes=num_classes)
    # samples, labels = swiss_roll(num_classes=num_classes)

    samples = torch.from_numpy(samples).float()
    labels = torch.from_numpy(labels).float()

    # normalize
    samples = (samples - samples.mean(axis=0)) / samples.std(axis=0)

    samples = torch.Tensor(samples).to(device)
    labels = torch.Tensor(labels).to(device)

    h0s = []
    h1s = []
    h0_nofs = []
    h1_nofs = []

    grid_preds = []
    grid_pred_logits_epoch = []

    model = SimpleLogisticRegression(activationFn, num_classes=num_classes)
    model = model.to(device)

    OPTIMIZER = torch.optim.AdamW(model.parameters(), lr=lr)
    CRITERION = torch.nn.NLLLoss().to(device)
    activationFn.__name__ = activationFn.__name__.lower()

    os.makedirs(f"out/{activationFn.__name__}", exist_ok=True)

    print(f"-------- {activationFn.__name__} ---------")

    s = plot_input_samples(activationFn, samples, labels)

    for epoch in range(num_epochs):
        OPTIMIZER.zero_grad()
        prediction = model(samples).squeeze()
        preds = torch.argmax(prediction, dim=1)
        acc = torch.sum(preds == labels) / len(labels)
        wrong_samples = preds != labels
        loss = CRITERION(prediction, labels.long())

        print(
            f"[{epoch}] loss: {loss.item():.2f}, Accuracy: {acc:.2f}, #Wrong: {sum(wrong_samples.int())}"
        )

        # ---- prepare weights ----
        f = model.activation
        w0 = model.linear1.weight.detach().cpu()
        b0 = model.linear1.bias.detach().cpu()

        w1 = model.linear2.weight.detach().cpu()
        b1 = model.linear2.bias.detach().cpu()

        x0 = samples.detach().cpu()
        xmin, xmax = x0[:, 0].min(), x0[:, 0].max()
        ymin, ymax = x0[:, 1].min(), x0[:, 1].max()

        # ---- Sample the space and classify each point in v
        #      giving us the predictions grid_preds with the probability associated
        #      with that particular point ----

        X, Y = np.meshgrid(np.linspace(xmin, xmax, r1), np.linspace(ymin, ymax, r2))
        v = (
            torch.stack(
                (torch.from_numpy(X.flatten()), torch.from_numpy(Y.flatten())), axis=1
            )
            .float()
            .to(device)
        )

        grid_pred_logits = model.forward(v).detach().cpu()
        grid_pred = torch.argmax(grid_pred_logits, dim=1)
        grid_pred = grid_pred.numpy()
        grid_preds.append(grid_pred)
        grid_pred_logits_epoch.append(grid_pred_logits)

        # ---- transformation from input -> layer1 -> layer2 -> output ----
        h0 = f((w0 @ x0.T) + b0[:, None]).T
        h1 = torch.nn.Softmax(dim=1)(f((w1 @ h0.T) + b1[:, None]).T)

        h0_nof = ((w0 @ x0.T) + b0[:, None]).T
        h1_nof = ((w1 @ h0.T) + b1[:, None]).T

        h0s.append(h0)
        h1s.append(h1)

        h0_nofs.append(h0_nof)
        h1_nofs.append(h1_nof)

        # Do the actual update step only after we've saved all the data
        # so that we can see iteration 0 with randomly initialized weights
        loss.backward()
        OPTIMIZER.step()

        plt.close("all")
        # plt.scatter(
        #     h0[:, 0],
        #     h0[:, 1],
        #     h0[:, 2],
        #     c=labels.cpu(),
        #     cmap="plasma",
        #     edgecolor="black",
        # )

        # plot every n epochs
        if epoch % 20 == 0:
            plot_boundary_with_inputs(
                activationFn,
                labels,
                r2,
                r1,
                s,
                epoch,
                X,
                Y,
                grid_pred_logits,
                False,
            )
            plot_contours_only(
                activationFn,
                r2,
                r1,
                epoch,
                X,
                Y,
                grid_pred_logits,
                False,
            )


if __name__ == "__main__":
    # activationFn = torch.nn.ReLU
    # activationFn = torch.nn.GELU
    # activationFn = torch.nn.Tanh
    # activationFn = torch.nn.Sigmoid
    # activationFn = torch.nn.LeakyReLU
    for activationFn in [
        # torch.nn.ReLU,
        # torch.nn.GELU,
        # torch.nn.Tanh,
        torch.nn.Sigmoid,
        # torch.nn.LeakyReLU,
    ]:
        torch.manual_seed(1337)

        main(activationFn)
        plt.close("all")
