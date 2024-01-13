import torch

# Define tensors with gradients
A = torch.tensor(2.0, requires_grad=True)
B = torch.tensor(-3.0, requires_grad=True)
C = torch.tensor(10.0, requires_grad=True)
D = torch.tensor(-2.0, requires_grad=True)

# Perform forward pass
E = A * B
F = C + E
L = D * F

# Retain gradients for intermediate variables
for var in [E, F, L]:
    var.retain_grad()

# List of variables and their names
vars = {
    "A": A,
    "B": B,
    "C": C,
    "D": D,
    "E": E,
    "F": F,
    "L": L,
}

# Backward pass
L.backward()

# Display values and gradients
for name, node in vars.items():
    print(f"| {name} | val:{node.item()} \t | grad:{node.grad.item()} \t|")
