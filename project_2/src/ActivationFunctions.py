import numpy as np

"""
Wrap all the activation functions in classes with static method, in that way we can simply feed
the class to the FeedForwardNeuralNet checking that the input class is a subclass of the
ActivationFunction interface, making it trivial to test new activation functions without
having to tweak the Neural network class.
"""


class ActivationFunction:
    @staticmethod
    def evaluate(x):
        pass

    @staticmethod
    def evaluate_derivative(x):
        pass

    def __repr__(self):
        pass


class Softmax(ActivationFunction):
    @staticmethod
    def evaluate(x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    @staticmethod
    def evaluate_derivative(x):
        # Not complete, this only gives da(zj)/dzj. Also needs cross-terms
        # da(zk)/dzj. Not used in current implementations.
        softmax_eval = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return softmax_eval - softmax_eval ** 2

    def __repr__(self):
        return "Softmax"


class ReLU(ActivationFunction):
    @staticmethod
    def evaluate(x):
        return np.maximum(0, x)

    @staticmethod
    def evaluate_derivative(x):
        return np.where(x < 0, 0, 1)

    def __repr__(self):
        return "ReLU"


class LeakyReLU(ActivationFunction):
    @staticmethod
    def evaluate(x):
        return np.where(x < 0, 0.1 * x, x)

    @staticmethod
    def evaluate_derivative(x):
        return np.where(x < 0, 0.1, 1)

    def __repr__(self):
        return "LeakyReLU"


class ID(ActivationFunction):
    @staticmethod
    def evaluate(x):
        return x

    @staticmethod
    def evaluate_derivative(x):
        return 1

    def __repr__(self):
        return "ID"


class ELU(ActivationFunction):
    @staticmethod
    def evaluate(x):
        return np.where(x < 0, np.exp(x) - 1, x)

    @staticmethod
    def evaluate_derivative(x):
        return np.where(x < 0, np.exp(x), 1)

    def __repr__(self):
        return "ELU"


class Sigmoid(ActivationFunction):
    @staticmethod
    def evaluate(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def evaluate_derivative(x):
        # sig(x) * (1 - sig(x))
        # compare with CostFunctions.CrossEntropy.evaluate_gradient
        # and note the cancelation with the substitution sig(x) <=> model
        return 1 / (1 + np.exp(-x)) * (1 - 1 / (1 + np.exp(-x)))

    def __repr__(self):
        return "Sigmoid"


class Tanh(ActivationFunction):
    @staticmethod
    def evaluate(x):
        return np.tanh(x)

    @staticmethod
    def evaluate_derivative(x):
        return 1 - np.tanh(x) ** 2

    def __repr__(self):
        return "Tanh"


if __name__ == "__main__":
    # quick check to ensure interfaces are implemented correctly
    for a in [ReLU, LeakyReLU, ELU, Sigmoid]:
        print(a())
        print(a.evaluate(10))
        print(a.evaluate_derivative(10), end="\n---\n")

    print(issubclass(ReLU, ActivationFunction))
