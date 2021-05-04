# This is a toy example showing the importance of data normalization
# Model: SimpleLogisticRegression is a model that consists of a linear and a sigmoid layer.
# Data: 2D points, each a gaussian centered around a different point
#      - there is an 'easy' dataset (which has 0 mean, unit std)
#      - there is a 'harder' dataset (does not have 0 mean, unit std)

import torch
import numpy as np
import matplotlib.pyplot as plt


class SimpleLogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        return self.sigmoid(x)


class DummyDataloader:
    valid_datasets = "easy hard".split(' ')

    def __init__(self, dataset=None):
        if dataset not in self.valid_datasets:
            raise ValueError(f"Valid Datasets are {self.valid_datasets}")
        data = torch.from_numpy(np.loadtxt(f"{dataset}.txt", delimiter=","))
        self.dataset_name = dataset
        self.num_samples = data.shape[0]
        self.samples = data[:, :-1].float()
        self.labels = data[:, -1].float()

    def get_data(self):
        return self.samples, self.labels


def plot_boundary_line(samples, labels, theta, dataset_name):
    c0 = samples[labels == 0]
    c1 = samples[labels == 1]
    plt.plot(c0[:, 0], c0[:, 1], 'r.')
    plt.plot(c1[:, 0], c1[:, 1], 'b+')
    plt.title(dataset_name)

    xmin, xmax = samples[:, 0].min(), samples[:, 0].max()
    ymin, ymax = samples[:, 1].min(), samples[:, 1].max()
    plot_x = np.array([xmin, xmax])
    plot_y = -(plot_x * theta[1] + theta[0]) / theta[2]
    plt.plot(plot_x, plot_y, 'y-', label='seperation', linewidth=3, alpha=0.4)
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])


if __name__ == '__main__':
    dl = DummyDataloader('easy')
    input_dim = 2
    output_dim = 1
    model = SimpleLogisticRegression(input_dim, output_dim)

    NUM_EPOCHS = 1000
    LR = 1
    SHOW_EVERY = 100
    OPTIMIZER = torch.optim.AdamW(model.parameters(), lr=LR)
    # OPTIMIZER = torch.optim.SGD(model.parameters(), lr=LR)
    CRITERION = torch.nn.BCELoss()

    samples = dl.get_data()[0]
    labels = dl.get_data()[1]

    for epoch in range(NUM_EPOCHS):
        OPTIMIZER.zero_grad()
        prediction = model(samples).squeeze()

        loss = CRITERION(prediction, labels)

        loss.backward()
        OPTIMIZER.step()

        if epoch % SHOW_EVERY == 0:
            b = model.linear.bias.detach().cpu()
            w = model.linear.weight.detach().cpu()
            l = torch.cat((torch.tensor([b]), w.flatten())).numpy()
            plot_boundary_line(samples, labels, l, dl.dataset_name + f' {epoch=}')

            plt.show(block=False)
            print(f'loss: {loss.item():.4f}')
            fig = plt.gcf()
            fig.canvas.draw()
            fig.canvas.flush_events()
    plt.show(block=True)
