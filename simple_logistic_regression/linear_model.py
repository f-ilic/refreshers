import torch
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data: x and y = ax^2 + bx + c + noise
np.random.seed(0)  # for reproducibility
x = np.linspace(-3, 3, 15)
y = (np.sin(1 * x) + 1) * 13
# y = (
#     # -2.2 * x**3
#     # + np.random.randn(*x.shape)
#     +1.5 * x**2
#     - 2 * x
#     + 1
#     + np.random.randn(*x.shape) * 3.4  # 5.5
# )

# Convert data to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Prepare the design matrix for a second-degree polynomial
# The design matrix is [x^2, x, 1]
X_design = torch.stack(
    [
        # x_tensor**12,
        # x_tensor**11,
        # x_tensor**10,
        # x_tensor**9,
        # x_tensor**8,
        # x_tensor**7,
        # x_tensor**6,
        # x_tensor**5,
        # x_tensor**4,
        # x_tensor**3,
        x_tensor**2,
        x_tensor,
        torch.ones_like(x_tensor),
    ],
    dim=1,
)

# Calculate weights using Moore-Penrose pseudo-inverse
weights = torch.pinverse(X_design) @ y_tensor.unsqueeze(1)

# Predict y using the calculated weights
y_pred = X_design @ weights

# Plotting
for i in range(len(x)):
    print(f"{x[i]:.2f} {y[i]:.2f}")

plt.scatter(x, y, label="Original data")
plt.plot(x, y_pred.detach().numpy(), color="red", label="Fitted curve")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Quadratic Polynomial Fitting using Moore-Penrose Pseudo-inverse")
plt.legend()
plt.show()

# print(weights.detach().numpy())  # Displaying the calculated weights (coefficients)

for i in range(len(weights)):
    print(f"{weights[i].item():.5f} * x^{len(weights) - i - 1} + ")
