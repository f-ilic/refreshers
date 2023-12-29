import torch
import numpy as np
import matplotlib.pyplot as plt
from generate_data import simple_dataset, donut_dataset, moon_dataset
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

torch.manual_seed(1337)


class SimpleLogisticRegression(torch.nn.Module):
    def __init__(self, activationFn=torch.nn.ReLU):
        super(SimpleLogisticRegression, self).__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(3, 3)
        self.linear3 = torch.nn.Linear(3, 10)
        self.linear4 = torch.nn.Linear(10, 3)

        self.activation = activationFn()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        x = self.activation(x)
        return self.softmax(x)


if __name__ == '__main__':
    # activationFns = [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.Tanh,  torch.nn.ELU, torch.nn.CELU, torch.nn.GELU]
    activationFns = [torch.nn.Tanh, torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.Sigmoid,  torch.nn.GELU]
    LRs = [0.14, 0.025, 0.025, 0.3, 0.03]

    all_axes = []


    # samples, labels = simple_dataset()
    samples, labels = donut_dataset(num_classes=3)
    # samples, labels = moon_dataset()

    samples = torch.from_numpy(samples).float()
    labels = torch.from_numpy(labels).float()

    # normalize
    samples = (samples - samples.mean(axis=0)) / samples.std(axis=0)

    # ---- Show the input data ----
    datafig, dataax = plt.subplots(1,len(activationFns)+1)
    X = np.linspace(-2, 2, 100)
    for id, fun in enumerate(activationFns):
        dataax[id + 1].plot(X, fun()(torch.Tensor(X)).numpy())
        dataax[id + 1].set_title(fun.__name__)
    datafig.set_size_inches(8, 1.5)
    datafig.tight_layout()

    for l in list(torch.unique(labels).int().numpy()):
        c = samples[labels == l, :]
        dataax[0].scatter(c[:,0], c[:,1])
    dataax[0].set_title("data to classify")

    samples = torch.Tensor(samples).cuda()
    labels = torch.Tensor(labels).cuda()

    d = {}
    for numfn, activationFn in enumerate(activationFns):
        model = SimpleLogisticRegression(activationFn)
        model = model.cuda()
        d[activationFn.__name__] = []

        NUM_EPOCHS = 5000
        SAVE_EVERY = 1000
        LR = LRs[numfn]
        OPTIMIZER = torch.optim.AdamW(model.parameters())#, lr=LR)
        # OPTIMIZER = torch.optim.SGD(model.parameters(), lr=LR)
        CRITERION = torch.nn.NLLLoss().cuda()
        print(f'-------- {activationFn.__name__} ---------')
        for epoch in range(NUM_EPOCHS):
            # ---- Actual training of model with different activation functions ----
            OPTIMIZER.zero_grad()
            prediction = model(samples).squeeze()
            preds = torch.argmax(prediction, dim=1)
            acc = torch.sum(preds == labels) / len(labels)
            wrong_samples = preds != labels
            loss = CRITERION(prediction, labels.long())

            # ---- COMPUTE ALL THE THINGS WE WANT TO VISUALIZE ----

            if epoch % SAVE_EVERY == 0:
                print(f'[{epoch}] loss: {loss.item():.2f}, Accuracy: {acc:.2f}, #Wrong: {sum(wrong_samples.int())}')

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
                #      giving us the predictions v_pred with the probability associated
                #      with that particular point ----
                r1 = r2 = 1000
                X, Y = np.meshgrid(np.linspace(xmin, xmax, r1), np.linspace(ymin, ymax, r2))
                v = torch.stack((torch.from_numpy(X.flatten()),
                                 torch.from_numpy(Y.flatten())), axis=1).float().cuda()
                v_pred = model.forward(v).detach().cpu()
                v_out = torch.argmax(v_pred, dim=1)
                v_pred = v_pred.numpy()

                # ---- transformation from input -> layer1 -> layer2 -> output ----
                ft_l, ft2_l, gt_l, pred_l = ([] for i in range(4))
                tmp_lbl = labels.cpu()
                tmp_pred = preds.cpu()

                for l in list(torch.unique(tmp_lbl).int().numpy()):
                    ft = f((w0 @ x0.T) + b0[:, None]).T
                    ft2 = f((w1 @ ft.T) + b1[:, None]).T
                    gt = x0[tmp_lbl == l, :]
                    ft = ft[tmp_pred == l, :]
                    pr = model.forward(samples)
                    ft2 = ft2[tmp_pred == l, :]
                    pred = x0[tmp_pred == l, :]

                    ft_l.append(ft.cpu())
                    ft2_l.append(ft2.cpu())
                    gt_l.append(gt.cpu())
                    pred_l.append(pred.cpu())

                d[activationFn.__name__].append({'ft_l': ft_l,
                                                  'ft2_l': ft2_l,
                                                  'gt_l': gt_l,
                                                  'pred_l': pred_l,
                                                  'v_pred': v_pred})

            # Do the actual update step only after we've saved all the data
            # so that we can see iteration 0 with randomly initialized weights
            loss.backward()
            OPTIMIZER.step()



    def update(val):
        epoch = int(epoch_slider.val)
        print(f'setting Epoch {epoch*SAVE_EVERY}')
        for numfn, activationFn in enumerate(activationFns):
            axes = ax[:, numfn]
            for a in axes:
                a.clear()

            epoch_dict = d[activationFn.__name__][epoch]

            v_pred = epoch_dict['v_pred']
            tmp = v_pred


            ft_l = epoch_dict['ft_l']
            ft2_l = epoch_dict['ft2_l']
            gt_l = epoch_dict['gt_l']
            pred_l = epoch_dict['pred_l']

            for ft, ft2, gt, pred in zip(ft_l, ft2_l, gt_l, pred_l):
                axes[0].scatter(ft[:, 0], ft[:, 1], ft[:, 2])
                axes[0].set_title(f'{activationFn.__name__}')
                axes[1].scatter(ft2[:, 0], ft2[:, 1], ft2[:, 2])
                axes[2].scatter(pred[:, 0], pred[:, 1])

                axes[3].contourf(X, Y, tmp[:,0].reshape((r1, r2)), cmap="RdYlBu")
                axes[4].contourf(X, Y, tmp[:,1].reshape((r1, r2)), cmap="RdYlBu")
                axes[5].contourf(X, Y, tmp[:,2].reshape((r1, r2)), cmap="RdYlBu")

        plt.show(block=False)

    fig, ax = plt.subplots(6, len(activationFns))
    fig.set_size_inches(13, 10)

    axepoch = plt.axes([0.2, 0.95, 0.65, 0.03])
    epoch_slider = Slider(axepoch, f'EPOCH * {SAVE_EVERY}', 0, (NUM_EPOCHS/SAVE_EVERY)-1, valinit=0, valstep=1)

    epoch_slider.on_changed(update)

    for i in range(len(activationFns)):
        ax[0][i].remove()
        ax[1][i].remove()
        ax[0][i] = fig.add_subplot(6, len(activationFns), i + 1, projection='3d')
        ax[1][i] = fig.add_subplot(6, len(activationFns), len(activationFns)+i+1, projection='3d')

    plt.show()

