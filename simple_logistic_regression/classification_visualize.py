import torch
import numpy as np
import matplotlib.pyplot as plt
from generate_data import simple_dataset, donut_dataset
from mpl_toolkits.mplot3d import Axes3D

torch.manual_seed(1337)

class SimpleLogisticRegression(torch.nn.Module):
    def __init__(self):
        super(SimpleLogisticRegression, self).__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(3, 2)
        self.activation = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        return self.softmax(x)

if __name__ == '__main__':
    model = SimpleLogisticRegression()

    NUM_EPOCHS = 100
    LR = 0.2
    OPTIMIZER = torch.optim.AdamW(model.parameters(), lr=LR)
    # OPTIMIZER = torch.optim.SGD(model.parameters(), lr=LR)
    CRITERION = torch.nn.NLLLoss()

    # samples, labels = simple_dataset()
    samples, labels = donut_dataset()

    samples = torch.from_numpy(samples).float()
    labels = torch.from_numpy(labels).long()

    # normalize
    # samples = (samples - samples.mean(axis=0)) / samples.std(axis=0)

    fig = plt.figure(figsize=[16, 8])
    ax0 = fig.add_subplot(2, 4, 1)
    ax1 = fig.add_subplot(2, 4, 2, projection='3d')
    ax2 = fig.add_subplot(2, 4, 3)
    ax3 = fig.add_subplot(2, 4, 4)

    ax4 = fig.add_subplot(2,4,8)

    for epoch in range(NUM_EPOCHS):
        OPTIMIZER.zero_grad()
        prediction = model(samples).squeeze()
        preds = torch.argmax(prediction, dim=1)

        acc = torch.sum(preds == labels)/len(labels)
        wrong_samples = preds != labels

        loss = CRITERION(prediction, labels)

        print(f'[{epoch}] loss: {loss.item():.2f}, Accuracy: {acc:.2f}, #Wrong: {sum(wrong_samples.int())}')

        loss.backward()
        OPTIMIZER.step()

    f = model.activation
    w0 = model.linear1.weight.detach().cpu()
    b0 = model.linear1.bias.detach().cpu()

    w1 = model.linear2.weight.detach().cpu()
    b1 = model.linear2.bias.detach().cpu()

    xmin, xmax = samples[:, 0].min(), samples[:, 0].max()
    ymin, ymax = samples[:, 1].min(), samples[:, 1].max()

    X, Y = np.meshgrid(np.linspace(xmin, xmax, 1000), np.linspace(ymin, ymax, 100))
    v = torch.stack((torch.from_numpy(X.flatten()),
                     torch.from_numpy(Y.flatten())), axis=1).float()
    v_pred = torch.argmax(model.forward(v), dim=1)

    for l in list(torch.unique(labels).int().numpy()):
        ft  = f((w0 @ samples.T) + b0[:,None]).T
        ft2 = f((w1 @ ft.T)      + b1[:,None]).T

        gt   = samples[labels == l, :]
        ft   = ft[preds == l, :]
        pr = model.forward(samples)
        ft2 = ft2[preds == l, :]
        pred = samples[preds  == l, :]
        grid = v[v_pred == l, :]

        ax0.scatter(gt[:, 0], gt[:, 1]);           ax0.set_title('Ground Truth Data')
        ax1.scatter(ft[:, 0], ft[:, 1], ft[:, 2]); ax1.set_title('layer1: 2x3 transform')
        ax2.scatter(ft2[:, 0], ft2[:, 1]);         ax2.set_title('layer2: 3x2 transform')
        ax3.scatter(pred[:, 0], pred[:, 1]);       ax3.set_title('output')

        ax4.scatter(grid[:, 0], grid[:, 1], s=2);       ax4.set_title('Sampled grid')

    plt.show()


