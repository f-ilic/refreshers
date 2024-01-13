import torch
import numpy as np
import matplotlib.pyplot as plt
from generate_data import simple_dataset, donut_dataset, moon_dataset, swiss_roll
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

torch.manual_seed(1337)

torch.pi = torch.acos(torch.zeros(1)).item() * 2

class SimpleLogisticRegression(torch.nn.Module):
    def __init__(self, activationFn=torch.nn.ReLU, num_classes=2):
        super(SimpleLogisticRegression, self).__init__()
        self.linear1 = torch.nn.Linear(2, 5)
        self.linear2 = torch.nn.Linear(5, num_classes)
        self.activation = activationFn()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        return self.softmax(x)


if __name__ == '__main__':

    # activationFn = torch.nn.ReLU
    activationFn = torch.nn.GELU
    
    lr = 0.85
    num_classes = 3
    num_epochs = 4200
    # samples, labels = donut_dataset(num_classes=num_classes)
    r1 = r2 = 60

    samples, labels = swiss_roll(num_classes=num_classes)

    samples = torch.from_numpy(samples).float()
    labels = torch.from_numpy(labels).float()

    # normalize
    samples = (samples - samples.mean(axis=0)) / samples.std(axis=0)


    samples = torch.Tensor(samples).cuda()
    labels = torch.Tensor(labels).cuda()

    h0s = []
    h1s = []
    grid_preds = []

    model = SimpleLogisticRegression(activationFn, num_classes=num_classes)
    model = model.cuda()

    OPTIMIZER = torch.optim.AdamW(model.parameters(), lr=lr)
    CRITERION = torch.nn.NLLLoss().cuda()
    print(f'-------- {activationFn.__name__} ---------')
    for epoch in range(num_epochs):
        # ---- Actual training of model with different activation functions ----
        OPTIMIZER.zero_grad()
        prediction = model(samples).squeeze()
        preds = torch.argmax(prediction, dim=1)
        acc = torch.sum(preds == labels) / len(labels)
        wrong_samples = preds != labels
        loss = CRITERION(prediction, labels.long())

        # ---- COMPUTE ALL THE THINGS WE WANT TO VISUALIZE ----

        print(f'[{epoch}] loss: {loss.item():.2f}, Accuracy: {acc:.2f}, #Wrong: {sum(wrong_samples.int())}')

        # ---- prepare weights ----
        f = model.activation
        w0 = model.linear1.weight.detach().cpu()
        b0 = model.linear1.bias.detach().cpu()

        w1 = model.linear2.weight.detach().cpu()
        b1 = model.linear2.bias.detach().cpu()

        x0 = samples.detach().cpu()
        xmin, xmax = x0[:, 0].min()-3, x0[:, 0].max()+3
        ymin, ymax = x0[:, 1].min()-3, x0[:, 1].max()+3

        # ---- Sample the space and classify each point in v
        #      giving us the predictions grid_preds with the probability associated
        #      with that particular point ----
        X, Y = np.meshgrid(np.linspace(xmin, xmax, r1), np.linspace(ymin, ymax, r2))
        v = torch.stack((torch.from_numpy(X.flatten()),
                            torch.from_numpy(Y.flatten())), axis=1).float().cuda()
        grid_pred_logits = model.forward(v).detach().cpu()
        grid_pred = torch.argmax(grid_pred_logits, dim=1)
        grid_pred = grid_pred.numpy()
        grid_preds.append(grid_pred)

        # ---- transformation from input -> layer1 -> layer2 -> output ----
        h0 = f((w0 @ x0.T) + b0[:, None]).T
        h1 = f((w1 @ h0.T) + b1[:, None]).T

        h0s.append(h0)
        h1s.append(h1)

        # Do the actual update step only after we've saved all the data
        # so that we can see iteration 0 with randomly initialized weights
        loss.backward()
        OPTIMIZER.step()



    def update(val):
        epoch = int(epoch_slider.val)
        axes = [ax1, ax2, ax3]
        for a in axes:
            a.clear()
        grid = grid_preds[val]
        h0 = h0s[epoch]
        h1 = h1s[epoch]
        ax1.scatter(h0[:,0], h0[:,1], h0[:,2], c=labels.cpu(), cmap='plasma')
        ax2.scatter(h1[:,0], h1[:,1], h1[:,2], c=labels.cpu(), cmap='plasma')
        ax3.contourf(X, Y, grid.reshape(r1,r2), cmap='plasma', alpha=0.7)
        ax3.contour(X, Y, grid.reshape(r1,r2), colors='k', linewidths=3, linestyles='dashed', levels=1)
        # ax3.contour(cs, colors='k')
        
        s = samples.cpu()
        ax3.scatter(s[:,0], s[:,1], c=labels.cpu(), cmap='plasma', edgecolor='black')
        # ax3.grid(c='k', ls='-', alpha=0.3)

        plt.show(block=False)

    fig = plt.figure(figsize=plt.figaspect(1/4.))
    ax0 = fig.add_subplot(1, 4, 1)
    ax1 = fig.add_subplot(1, 4, 2, projection='3d')
    ax2 = fig.add_subplot(1, 4, 3, projection='3d')
    ax3 = fig.add_subplot(1, 4, 4)


    # ---- Show the input data ----
    s = samples.cpu()
    ax0.scatter(s[:,0], s[:,1], c=labels.cpu(), cmap='plasma', edgecolor='black')

    axepoch = plt.axes([0.2, 0.95, 0.65, 0.03])
    epoch_slider = Slider(axepoch, f'epoch', 0, len(grid_preds)-1, valinit=0, valstep=1)

    epoch_slider.on_changed(update)


    plt.show(block=True)

