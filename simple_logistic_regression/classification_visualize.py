import torch
import numpy as np
import matplotlib.pyplot as plt
from generate_data import simple_dataset, donut_dataset
from mpl_toolkits.mplot3d import Axes3D

torch.manual_seed(1337)


class SimpleLogisticRegression(torch.nn.Module):
    def __init__(self, activationFn=torch.nn.ReLU):
        super(SimpleLogisticRegression, self).__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(3, 2)
        self.activation = activationFn()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        return self.softmax(x)


if __name__ == '__main__':

    activationFns = [torch.nn.ReLU, torch.nn.Tanh, torch.nn.Sigmoid, torch.nn.ELU]
    # activationFns_LR = []
    all_axes = []

    fig, ax = plt.subplots(3, len(activationFns))
    fig.set_size_inches(13,10)

    # samples, labels = simple_dataset()
    samples, labels = donut_dataset()

    samples = torch.from_numpy(samples).float()
    labels = torch.from_numpy(labels).long()

    # normalize
    samples = (samples - samples.mean(axis=0)) / samples.std(axis=0)

    # ---- Show the input data ----
    # datafig = plt.figure()
    # dataax = datafig.add_subplot(1,1,1)
    # for l in list(torch.unique(labels).int().numpy()):
    #     c = samples[labels == l, :]
    #     dataax.scatter(c[:,0], c[:,1])
    # dataax.set_title("data to classify")

    # ---- Set up the figure ----
    for i in range(len(activationFns)):
        ax[0][i].remove()
        ax[0][i] = fig.add_subplot(4, len(activationFns), i+1, projection='3d')


    for numfn, activationFn in enumerate(activationFns):
        axes = ax[:, numfn]
        model = SimpleLogisticRegression(activationFn)

        NUM_EPOCHS = 1
        LR = 0.01
        OPTIMIZER = torch.optim.AdamW(model.parameters(), lr=LR)
        # OPTIMIZER = torch.optim.SGD(model.parameters(), lr=LR)
        CRITERION = torch.nn.NLLLoss()

        # ---- Actual training of model with different activation functions ----
        for epoch in range(NUM_EPOCHS):
            OPTIMIZER.zero_grad()
            prediction = model(samples).squeeze()
            preds = torch.argmax(prediction, dim=1)
            acc = torch.sum(preds == labels) / len(labels)
            wrong_samples = preds != labels
            loss = CRITERION(prediction, labels)
            loss.backward()
            OPTIMIZER.step()
        print(f'[{epoch}] loss: {loss.item():.2f}, Accuracy: {acc:.2f}, #Wrong: {sum(wrong_samples.int())}')

        # ---- prepare weights ----
        f = model.activation
        w0 = model.linear1.weight.detach().cpu()
        b0 = model.linear1.bias.detach().cpu()

        w1 = model.linear2.weight.detach().cpu()
        b1 = model.linear2.bias.detach().cpu()

        xmin, xmax = samples[:, 0].min(), samples[:, 0].max()
        ymin, ymax = samples[:, 1].min(), samples[:, 1].max()

        # ---- Sample the space and classify each point in v
        #      giving us the predictions v_pred with the probability associated
        #      with that particular point ----
        r1 = r2 = 1000
        X, Y = np.meshgrid(np.linspace(xmin, xmax, r1), np.linspace(ymin, ymax, r2))
        v = torch.stack((torch.from_numpy(X.flatten()),
                         torch.from_numpy(Y.flatten())), axis=1).float()
        v_pred = model.forward(v).detach().cpu()
        v_out = torch.argmax(v_pred, dim=1)
        v_pred = v_pred.numpy()
        tmp = v_pred[:, 0]
        contour = axes[2].contourf(X, Y, tmp.reshape((r1, r2)), cmap="RdYlBu", vmin=0, vmax=1)#, levels=np.linspace(0, 1, 11))
        ax_c = fig.colorbar(contour, ax=axes[2])
        ax_c.set_label(f"$P(y = 0)$")

        # ---- Plot the transformation from input -> layer1 -> layer2 -> output ----
        for l in list(torch.unique(labels).int().numpy()):
            ft = f((w0 @ samples.T) + b0[:, None]).T
            ft2 = f((w1 @ ft.T) + b1[:, None]).T

            gt = samples[labels == l, :]
            ft = ft[preds == l, :]
            pr = model.forward(samples)
            ft2 = ft2[preds == l, :]
            pred = samples[preds == l, :]

            axes[0].scatter(ft[:, 0], ft[:, 1], ft[:, 2])
            axes[0].set_title(f'{activationFn.__name__}')
            axes[1].scatter(ft2[:, 0], ft2[:, 1])
            axes[2].scatter(pred[:, 0], pred[:, 1])


    fig.tight_layout()
    plt.show()
