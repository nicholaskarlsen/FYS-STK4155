import numpy as np

"""
Various cost functions and their gradients. Follows a general interface of (X, y, predictors, *args) where X denotes the design matrix, y the response.

"""

# Define a clear interace for all the Cost functions (Primarily for the NN class)
class CostFunction:
    def evaluate(X, y, predictors):
        pass

    def evaluate_gradient(X, y, predictors):
        pass

    def __repr__(self):
        pass

class OLS(CostFunction):
    def evaluate(X, y, predictors):
        return (y - X @ predictors).T @ (y - X @ predictors)

    def evaluate_gradient(X, y, predictors):
        return -2 * X.T @ (y - X @ predictors)

    def __repr__(self):
        return "OLS"


class Ridge(CostFunction):
    def evaluate(X, y, predictors, lambd):
        return OLS.evaluate(X, y, predictors) + lamb * predictors.T @ predictors

    def evaluate_gradient(X, y, predictors, lambd):
        return OLS.evaluate_gradient(X, y, predictors) + 2 * lambd * predictors

    def __repr__(self):
        return "Ridge"


# Consider depricating the old functions later, but keep for now due to existing SGD code

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
