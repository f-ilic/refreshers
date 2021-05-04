# This is a toy example showing the importance of data normalization
# Model: SimpleLogisticRegression is a model that consists of a linear and a sigmoid layer.
# Data: 2D points, each a gaussian centered around a different point
#      - there is an 'easy' dataset (which has 0 mean, unit std)
#      - there is a 'harder' dataset (does not have 0 mean, unit std)

import torch
import numpy as np
import matplotlib.pyplot as plt
from generate_data import generate_donut_and_gaussian, multiple_blobs_dataset
from mpl_toolkits.mplot3d import Axes3D


class SimpleLogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLogisticRegression, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, 3)
        self.linear2 = torch.nn.Linear(3, 2)
        self.linear3 = torch.nn.Linear(2, 1)
        self.activation = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)

        x = self.linear3(x)
        x = self.sigmoid(x)
        return x


def plot_boundary_line(ax, samples, labels, theta, dataset_name):
    c0 = samples[labels == 0]
    c1 = samples[labels == 1]
    ax.plot(c0[:, 0], c0[:, 1])
    ax.plot(c1[:, 0], c1[:, 1])
    ax.set_title(dataset_name)

    xmin, xmax = samples[:, 0].min(), samples[:, 0].max()
    ymin, ymax = samples[:, 1].min(), samples[:, 1].max()
    plot_x = np.array([xmin, xmax])
    plot_y = -(plot_x * theta[1] + theta[0]) / theta[2]
    ax.plot(plot_x, plot_y, 'y-', label='seperation', linewidth=3)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])


if __name__ == '__main__':
    input_dim = 2
    output_dim = 1
    model = SimpleLogisticRegression(input_dim, output_dim)

    NUM_EPOCHS = 1000
    LR = 0.4
    SHOW_EVERY = 1000
    OPTIMIZER = torch.optim.AdamW(model.parameters(), lr=LR)
    # OPTIMIZER = torch.optim.SGD(model.parameters(), lr=LR)
    CRITERION = torch.nn.BCELoss()

    samples, labels = generate_donut_and_gaussian()
    # samples, labels = multiple_blobs_dataset()
    samples = torch.from_numpy(samples).float()
    labels = torch.from_numpy(labels).float()
    fig = plt.figure()
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.set_title("Input Data")
    ax1 = fig.add_subplot(1, 3, 2, projection='3d')
    ax1.set_title("After first Layer f((3x2)*(2xN)+b)")
    ax2 = fig.add_subplot(1, 3, 3)
    ax2.set_title("After second Layer f((2x3)*(3xN)+b)")
    for epoch in range(NUM_EPOCHS):
        OPTIMIZER.zero_grad()
        prediction = model(samples).squeeze()
        acc = torch.sum((prediction > 0.5).int() == labels)/len(labels)
        loss = CRITERION(prediction, labels)

        print(f'loss: {loss.item():.2f}, Accuracy: {acc:.2f}')

        loss.backward()
        OPTIMIZER.step()

        if epoch % SHOW_EVERY == 0:
            # plt.show(block=False)
            T1 = model.linear1.weight.detach().cpu()
            b1 = model.linear1.bias.detach().cpu().T

            T2 = model.linear2.weight.detach().cpu()
            b2 = model.linear2.bias.detach().cpu().T

            f = model.activation
            sig = model.sigmoid

            x0 = samples.T
            x1 = f((T1 @ x0) + b1[:, None])
            x2 = f((T2 @ x1) + b2[:, None])

            for l in list(torch.unique(labels).int().numpy()):
                ax0.scatter(x0[:,labels == l][0, :], x0[:,labels == l][1, :])
                ax1.scatter(x1[:,labels == l][0, :], x1[:,labels == l][1, :],x1[:,labels == l][2, :])
                ax2.scatter(x2[:,labels == l][0, :], x2[:,labels == l][1, :])


            fig = plt.gcf()
            fig.canvas.draw()
            fig.canvas.flush_events()
    plt.show(block=True)
