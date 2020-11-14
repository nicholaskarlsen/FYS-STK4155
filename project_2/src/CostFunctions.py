import numpy as np


"""
Define a clear interace for all the Cost functions (Primarily for the NN class)
"""


class CostFunction:
    def evaluate(model, data):
        pass

    def evaluate_gradient(model, data):
        pass

    def __repr__(self):
        pass


class SquareError(CostFunction):
    def evaluate(model, data):
        return np.sum((model - data) ** 2) / 2

    def evaluate_gradient(model, data):
        return model - data

    def __repr__(self):
        return "OLS"


class CrossEntropy(CostFunction):
    def evaluate(model, data):
        # model => a
        # data => Y
        return -np.sum(data * np.log(model) + (1 - data) * np.log(1 - model))

        # model = [M, n]

    def evaluate_gradient(model, data):
        # Note the cancelation when multiplied with derivative of
        # activation if model is sigmoid.
        return -(data - model) / (model - model ** 2)

    def __repr__(self):
        return "Cross Entropy"


class Softmax_loss(CostFunction):
    def evaluate(model, data):
        # model => a
        # data => Y
        return -np.sum(data * np.log(model))

        # model = [M, n]

    def evaluate_gradient(model, data):
        # Not complete. Unused in current implementation. When using softmax
        # as activation, dcost/dzj = sum_k dcost/da_k *da_k/dz_j is no longer
        # zero for k != j, and thus not amenable to this simple treatment.
        return None  # - (data - model) / (model - model**2)

    def __repr__(self):
        return "Cross Entropy"


# KEEP THIS FOR SGD


def OLS_cost(X, y, predictors):
    return (y - X @ predictors).T @ (y - X @ predictors)


def OLS_cost_gradient(X, y, predictors):
    return -2 * X.T @ (y - X @ predictors)


def Ridge_cost(X, y, predictors, lambd):
    return OLS_cost(X, y, predictors) + lamb * predictors.T @ predictors


def Ridge_cost_gradient(X, y, predictors, lambd):
    return OLS_cost_gradient(X, y, predictors) + 2 * lambd * predictors
