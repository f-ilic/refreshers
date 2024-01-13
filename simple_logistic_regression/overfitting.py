import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Step 1: Generate synthetic data
x = torch.linspace(-5, 5, 10).unsqueeze(1)
y = 1.0 + 2.0 * x + 3.0 * x**2 + torch.randn(x.size()) * 44.5 - 3.0 * x**3


# Step 2: Define a More Complex Polynomial Model
class ComplexPolynomialModel(nn.Module):
    def __init__(self):
        super(ComplexPolynomialModel, self).__init__()
        self.fc1 = nn.Linear(1, 30)
        self.fc2 = nn.Linear(30, 100)
        self.fc3 = nn.Linear(100, 1100)
        self.fc4 = nn.Linear(1100, 10)
        self.fc5 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


model = ComplexPolynomialModel()

# Step 3: Define Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.01)

# Step 4: Training Loop
for epoch in range(300):  # More epochs to increase the chance of overfitting
    pred_y = model(x)
    loss = criterion(pred_y, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.item())

# Plotting the results
plt.scatter(x.numpy(), y.numpy(), label="Original Data")
x = torch.linspace(-5, 5, 120).unsqueeze(1)

with torch.no_grad():  # We don't need to track gradients for plotting
    plt.plot(x.numpy(), model(x).numpy(), label="Fitted Polynomial", color="red")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Polynomial Fit Demonstrating Overfitting")
plt.show()
