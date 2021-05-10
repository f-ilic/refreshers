import numpy as np
import matplotlib.pyplot as plt
from skimage import data

np.random.seed(1)

n = 3
m = 5
A = (np.random.rand(n,m)*10).astype(int)

# Identity operator
I_op = np.eye(n*m).astype(int)
AxI_op = np.reshape(I_op@A.flatten(), (n,m))
print(f"A == A @ I_op:\n{AxI_op == A}")


# gradient in X direction operator
A = data.camera()
Adx = np.gradient(A, axis=1)
print(A)
print(Adx)


fig, ax = plt.subplots(1,2)
ax[0].imshow(A, cmap='gray')
ax[1].imshow(Adx, cmap='gray')
plt.show()