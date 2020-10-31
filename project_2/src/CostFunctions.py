import numpy as np

"""
Various cost functions and their gradients. Follows a general interface of (X, y, predictors, *args) where X denotes the design matrix, y the response.

"""

# Define a clear interace for all the Cost functions (Primarily for the NN class)
class CostFunction:
    def evaluate(model, data):
        pass

    def evaluate_gradient(model, data):
        pass

    def __repr__(self):
        pass

# CONSIDER THIS CAREFULLY

class SquareError(CostFunction):
    def evaluate(model, data):
        return (data - model)**2 / 2

    def evaluate_gradient(model, data):
        return model - data

    def __repr__(self):
        return "OLS"

# CONSIDER THIS CAREFULLY

class Ridge(CostFunction):
    def evaluate(model, data):
        return

    def evaluate_gradient(model, data):
        return
    def __repr__(self):
        return "Ridge"


# KEEP THIS FOR SGD


def OLS_cost(X, y, predictors):
    return (y - X @ predictors).T @ (y - X @ predictors)


def OLS_cost_gradient(X, y, predictors):
    return -2 * X.T @ (y - X @ predictors)


def Ridge_cost(X, y, predictors, lambd):
    return OLS_cost(X, y, predictors) + lamb * predictors.T @ predictors


def Ridge_cost_gradient(X, y, predictors, lambd):
    return OLS_cost_gradient(X, y, predictors) + 2 * lambd * predictors


"""
# USE *args instead of lambdas!
def foo(x, f, *lamb):
    f(x, *lamb)

def bar1(x):
    print(x)

def bar2(x, lamb):
    print(x, lamb)

foo(1, bar1)
foo(1, bar2, 1)
"""
