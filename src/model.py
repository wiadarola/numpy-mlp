import numpy as np
from numpy.typing import NDArray

from src import functional as F


class NumpyMLP:
    def __init__(self, hidden_dim_1: int, hidden_dim_2: int):
        """
        :param hidden_dim_1: Size of hidden layer 1
        :param hidden_dim_2: Size of hidden layer 2
        """
        self.w1 = np.random.uniform(0, 1, (784, hidden_dim_1))
        self.b1 = np.zeros(hidden_dim_1)

        self.w2 = np.random.uniform(0, 1, (hidden_dim_1, hidden_dim_2))
        self.b2 = np.zeros(hidden_dim_2)

        self.w3 = np.random.uniform(0, 1, (hidden_dim_2, 10))
        self.b3 = np.zeros(10)

        self.cache: dict[str, NDArray]

    def __call__(self, x: NDArray) -> NDArray:
        return self.forward(x)

    def forward(self, x: NDArray) -> NDArray:
        B = x.shape[0]
        x = x.reshape((B, -1))

        z1 = x @ self.w1 + self.b1
        a1 = F.sigmoid(z1)

        z2 = a1 @ self.w2 + self.b2
        a2 = F.sigmoid(z2)

        z3 = a2 @ self.w3 + self.b3
        y_hat = F.softmax(z3)

        self.cache = {"z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3}

        return y_hat

    def backward(self, x: NDArray, y: NDArray, lr: float):
        """Compute gradients and update model parameters"""
        B = x.shape[0]
        x = x.reshape(B, -1)
        z1, a1, z2, a2, z3 = self.cache.values()

        y_one_hot = np.zeros((B, 10))
        y_one_hot[range(B), y] = 1.0

        dL_dz3 = F.softmax(z3) - y_one_hot  # C 1

        dL_da2 = dL_dz3 @ self.w3.T  # B H2
        dL_dz2 = dL_da2 * F.sigmoid_dz(z2)  # B H2

        dL_da1 = dL_dz2 @ self.w2.T  # B H1
        dL_dz1 = dL_da1 * F.sigmoid_dz(z1)  # H1 1

        # Parameter gradients
        dL_dw3 = np.expand_dims(a2, 2) @ np.expand_dims(dL_dz3, 1)  # H2 C
        dL_db3 = dL_dz3  # C

        dL_dw2 = np.expand_dims(a1, 2) @ np.expand_dims(dL_dz2, 1)  # H1 H2
        dL_db2 = dL_dz2  # H2

        dL_dw1 = np.expand_dims(x, 2) @ np.expand_dims(dL_dz1, 1)  # D H1
        dL_db1 = dL_dz1  # H1

        # Gradient step
        self.w1 = self.w1 - lr * dL_dw1.mean(0)
        self.b1 = self.b1 - lr * dL_db1.mean(0)
        self.w2 = self.w2 - lr * dL_dw2.mean(0)
        self.b2 = self.b2 - lr * dL_db2.mean(0)
        self.w3 = self.w3 - lr * dL_dw3.mean(0)
        self.b3 = self.b3 - lr * dL_db3.mean(0)
