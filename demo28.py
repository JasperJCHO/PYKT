from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

X = np.r_[np.random.randn(50, 2) + [2, 2],
          np.random.randn(50, 2) + [0, -2],
          np.random.randn(50, 2) + [-2, 2]]
print(X.shape)
[plt.scatter(e[0], e[1], c='black', s=7) for e in X]

k = 3
C_x = np.random.uniform(np.min(X[:, 0]), np.max(X[:, 0]), size=k)
C_y = np.random.uniform(np.min(X[:, 1]), np.max(X[:, 1]), size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
plt.scatter(C_x, C_y, marker='*', s=200, c='#C0FFEE')
plt.show()