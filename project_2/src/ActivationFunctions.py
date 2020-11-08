import numpy as np

# Wrap all the activation functions in classes with static methods


class ActivationFunction:
    def evaluate(x):
        pass

    def evaluate_derivative(x):
        pass

    def __repr__(self):
        pass

class Softmax(ActivationFunction):
    def evaluate(x):
        return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)

    def evaluate_derivative(x):
        softmax_eval = np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)
        return softmax_eval - softmax_eval**2

    def __repr__(self):
        return "Softmax"

class ReLU(ActivationFunction):
    def evaluate(x):
        return np.maximum(0, x)

    def evaluate_derivative(x):
        return np.where(x < 0, 0, 1)

    def __repr__(self):
        return "ReLU"


class LeakyReLU(ActivationFunction):
    def evaluate(x):
        return np.where(x < 0, 0.1 * x, x)

    def evaluate_derivative(x):
        return np.where(x < 0, 0.1, 1)

    def __repr__(self):
        return "LeakyReLU"

class ID(ActivationFunction):
    def evaluate(x):
        return x

    def evaluate_derivative(x):
        return 1

    def __repr__(self):
        return "ID"


class ELU(ActivationFunction):
    def evaluate(x):
        return np.where(x < 0, np.exp(x) - 1, x)

    def evaluate_derivative(x):
        return np.where(x < 0, np.exp(x), 1)

    def __repr__(self):
        return "ELU"


class Sigmoid(ActivationFunction):
    def evaluate(x):
        return 1 / (1 + np.exp(-x))

    def evaluate_derivative(x):
        # sig(x) * (1 - sig(x))
        return 1 / (1 + np.exp(-x)) * (1 - 1 / (1 + np.exp(-x)))

    def __repr__(self):
        return "Sigmoid"


class Tanh(ActivationFunction):
    def evaluate(x):
        return np.tanh(x)

    def evaluate_derivative(x):
        return 1 - Tanh(x) ** 2

    def __repr__(self):
        return "Tanh"


if __name__ == "__main__":
    # quick check to ensure interfaces are implemented correctly
    for a in [ReLU, LeakyReLU, ELU, Sigmoid]:
        print(a())
        print(a.evaluate(10))
        print(a.evaluate_derivative(10), end="\n---\n")

    print(issubclass(ReLU, ActivationFunction))
