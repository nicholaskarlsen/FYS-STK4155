import numpy as np


def ReLU(x):
    return np.maximum(0, x)


def ReLU_derivative(x):
    return np.where(x < 0, 0, 1)


def LeakyReLU(x):
    return np.where(x < 0, 0.1 * x, x)


def LeakyReLU_derivative(x):
    return np.where(x < 0, 0.1, 1)


def ELU(x):
    return np.where(x < 0, np.exp(x) - 1, x)


def ELU_derivative(x):
    return np.where(x < 0, np.exp(x), 1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


if __name__ == "__main__":
    print("Hello")
    print(ActivationFunction(123))
