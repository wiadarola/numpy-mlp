import numpy as np
from numpy.typing import NDArray


def categorical_cross_entropy(y: NDArray, y_hat: NDArray) -> NDArray:
    return (-np.log(y_hat + 1e-15))[range(y.shape[0]), y].mean()


def accuracy_score(y: NDArray, y_hat: NDArray):
    return np.count_nonzero(y == y_hat.argmax(1)) / len(y)


def sigmoid(z: NDArray) -> NDArray:
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_dz(z: NDArray) -> NDArray:
    return sigmoid(z) * (1.0 - sigmoid(z))


def softmax(z: NDArray) -> NDArray:
    exp_z = np.exp(z - z.max())
    return exp_z / exp_z.sum()
