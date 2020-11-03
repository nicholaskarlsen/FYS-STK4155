import numpy as np
import matplotlib.pyplot as plt
import numba
import sys

import CostFunctions

sys.path.insert(0, "../../project_1/src")
from FrankeFunction import *
import linear_regression


def minibatch(x, M):
    """Splits data set x into M roughly equally minibatches. If not evenly divisible, the excess
    is evenly spread throughout some of the batches.

    Args:
        x (Array): Data set
        M (Int): Number of minibatches

    Returns:
        Array: [M,.]-dim array containing the minibatch indices
    """
    indices = np.random.permutation(len(x))  # random permutation of [0, ..., len(x)-1]
    indices = np.array_split(indices, M)  # Split permutation into M sub-arrays
    return indices


# Stochastic Gradient Descent
def SGD(x, y, M, init_w, n_epochs, learning_rate, cost_gradient, *lambd):
    """Performs Stochastic Gradient Descent (SGD) to optimize a cost
    function given its gradient.

    Args:
        x (Array)               : Design matrix
        y (Array)               : response variable
        M (Int)                 : Number of mini-batches to split data set into
        init_w (Array)          : Initial/starting weights
        n_epochs (int)          : Number of epochs
        learning_rate (Float)   :
        cost_gradient (Func): Gradient of the cost function to optimize
        lambd (Float)           : Optional penalty parameter that goes into cost_gradient
    Returns:
        w (Array)               : Optimal weights corresponding to the cost gradient
    """
    w = init_w
    for epoch in range(n_epochs):
        mb = minibatch(x, M)  # Split x into M minibatches
        for i in range(M):
            # Pick out a random mini-batch index
            # compute gradient with random minibatch
            grad = cost_gradient(x[mb[i]], y[mb[i]], w, *lambd) / x[mb[i]].shape[0]
            # increment weights
            w = w - learning_rate * grad
    return w


# Stochastic Gradient Descent with Momentum (SGDM)
def SGDM(x, y, M, init_w, n_epochs, learning_rate, momentum, cost_gradient, *lambd):
    """Performs Stochastic Gradient Descent with Momentum (SGDM) to optimize a cost
    function given its gradient.

    Args:
        x (Array)               : Design matrix
        y (Array)               : response variable
        M (Int)                 : Number of mini-batches to split data set into
        init_w (Array)          : Initial/starting weights
        n_epochs (int)          : Number of epochs
        learning_rate (Float)   :
        momentum (Float)        : Momentum pramameter in the SGDM method, must be in [0,1]
        cost_gradient (Func)    : Gradient of the cost function to optimize
        lambd (Float)           : Optional penalty parameter that goes into cost_gradient
    Returns:
        w (Array)               : Optimal weights corresponding to the cost gradient
    """
    mb = minibatch(x, M)  # Split x into M minibatches
    w = init_w
    dw = 0

    for epoch in range(n_epochs):
        for i in range(M):
            # Pick out a random mini-batch index
            k = np.random.randint(M)
            # compute gradient with random minibatch
            grad = cost_gradient(x[mb[k]], y[mb[k]], w, *lambd) / x[mb[k]].shape[0]
            # increment weights
            dw = momentum * dw - learning_rate * grad
            w = w + dw
    return w


def ADAgrad(x, y, M, init_w, n_epochs, learning_rate, cost_gradient, *lambd):
    """Performs the Adaptive gradient algorithm (ADAGrad) to optimize a cost
    function given its gradient.

    Args:
        x (Array)               : Design matrix
        y (Array)               : response variable
        M (Int)                 : Number of mini-batches to split data set into
        init_w (Array)          : Initial/starting weights
        n_epochs (int)          : Number of epochs
        learning_rate (Float)   :
        cost_gradient (Func)    : Gradient of the cost function to optimize
        lambd (Float)           : Optional penalty parameter that goes into cost_gradient
    Returns:
        w (Array)               : Optimal weights corresponding to the cost gradient
    """
    mb = minibatch(x, M)
    w = init_w
    g_ti = np.zeros(len(init_w))
    for epoch in range(n_epochs):
        for i in range(M):
            # Pick out a random mini-batch index
            k = np.random.randint(M)
            # compute gradient with random minibatch
            grad = cost_gradient(x[mb[k]], y[mb[k]], w, *lambd) / x[mb[k]].shape[0]
            g_ti += np.dot(grad, grad)  # Gradient squared
            w = w - learning_rate / np.sqrt(g_ti) * grad

    return w


def RMSprop(x, y, M, init_w, n_epochs, learning_rate, forgetting_factor, cost_gradient, *lambd):
    """Performs the Adaptive gradient algorithm (ADAGrad) to optimize a cost
    function given its gradient.

    Args:
        x (Array)               : Design matrix
        y (Array)               : response variable
        M (Int)                 : Number of mini-batches to split data set into
        init_w (Array)          : Initial/starting weights
        n_epochs (int)          : Number of epochs
        learning_rate (Float)   :
        forgetting_factor(Float):
        cost_gradient (Func)    : Gradient of the cost function to optimize
        lambd (Float)           : Optional penalty parameter that goes into cost_gradient
    Returns:
        w (Array)               : Optimal weights corresponding to the cost gradient
    """
    mb = minibatch(x, M)
    w = init_w
    v = 0
    for epoch in range(n_epochs):
        for i in range(M):
            # Pick out a random mini-batch index
            k = np.random.randint(M)
            # compute gradient with random minibatch
            grad = cost_gradient(x[mb[k]], y[mb[k]], w, *lambd) / x[mb[k]].shape[0]
            v = forgetting_factor * v + (1 - forgetting_factor) * np.dot(grad, grad)
            w = w - learning_rate / np.sqrt(v) * grad

    return w


if __name__ == "__main__":
    np.random.seed(123)
    x = np.random.uniform(0, 1, 500)
    y = np.random.uniform(0, 1, 500)
    z = FrankeFunction(x, y)

    deg = 2

    X = linear_regression.design_matrix_2D(x, y, deg)
    N_predictors = int((deg + 1) * (deg + 2) / 2)
    w_init = np.random.randn(N_predictors)

    w_SGD_OLS = SGD(
        X,
        z,
        M=250,
        init_w=w_init,
        n_epochs=100,
        learning_rate=0.01,
        cost_gradient=CostFunctions.OLS_cost_gradient,
    )
    w_SGDM_OLS = SGDM(
        X,
        z,
        M=250,
        init_w=w_init,
        n_epochs=100,
        learning_rate=0.01,
        momentum=0.5,
        cost_gradient=CostFunctions.OLS_cost_gradient,
    )
    w_ADAgrad_OLS = ADAgrad(
        X,
        z,
        M=250,
        init_w=w_init,
        n_epochs=100,
        learning_rate=1,
        cost_gradient=CostFunctions.OLS_cost_gradient,
    )
    w_RMSProp_OLS = RMSprop(
        X,
        z,
        M=250,
        init_w=w_init,
        n_epochs=100,
        learning_rate=0.01,
        forgetting_factor=0.9,
        cost_gradient=CostFunctions.OLS_cost_gradient,
    )

    # w_SGD_Ridge = SGD(X, z, 250, w_init, 100, 0.01, CostFunctions.Ridge_cost_gradient, 0)
    # w_SGDM_Ridge = SGDM(X, z, 250, w_init, 100, 0.01, 0.5, CostFunctions.Ridge_cost_gradient, 0)

    w_analytic = linear_regression.OLS_2D(X, z)

    print(w_SGD_OLS, "- SGD OLS")
    print(w_SGDM_OLS, "- SGDM OLS")
    print(w_ADAgrad_OLS, "- ADAgrad OLS")
    print(w_RMSProp_OLS, "- RMSprop OLS")
    print(w_analytic, "- OLS")

    """
    for m in [2, 5, 10, 50]:
        print("\nm=%i" % m)
        for n in [10000]:
            w = SGD(X, z, m, np.random.randn(int((deg+1)*(deg+2)/2)), n, )
            print("epochs = %i" % n, w)
        print(w_analytic)
        print("diff = ", abs(w_analytic - w))
    """

    # w = SGD(X, z, 50, np.random.randn(int((deg+1)*(deg+2)/2)), 1000, 0.1, 100)
