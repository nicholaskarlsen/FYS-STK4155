import numpy as np
import matplotlib.pyplot as plt
import numba

import sys
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

#@numba.jit(nopython=True)
def SGD(x, y, M, w, n_epochs, t0, t1):
    mb = minibatch(x, M)
    for epoche in range(n_epochs):
        for i in range(M):
            k = np.random.randint(M)
            xi = x[mb[k]]
            yi = y[mb[k]]
            grad = - 2 * xi.T @ (yi - xi @ w)
            t = n_epochs * M + i
            w = w - (t0 / (t + t1)) * grad

    return w

    
def gamma(t0, t1, n_epochs, M):
    learning_rate = np.zeros(n_epochs * M)
    for e in range(n_epochs):
        for i in range(M):
            t = e * M + i
            learning_rate[t] = t0 / (t + t1)
    return learning_rate


if __name__ == "__main__":
    np.random.seed(123)
    x = np.random.uniform(0,1,500)
    y = np.random.uniform(0,1,500)
    z = FrankeFunction(x, y)
    
    deg = 5    

    X = linear_regression.design_matrix_2D(x, y, deg)
    w_analytic = linear_regression.OLS_2D(X, z)

    """
    for m in [2, 5, 10, 50]:
        print("\nm=%i" % m)
        for n in [10000]:
            w = SGD(X, z, m, np.random.randn(int((deg+1)*(deg+2)/2)), n, )
            print("epochs = %i" % n, w)
        print(w_analytic)
        print("diff = ", abs(w_analytic - w))
    """
    
    #w = SGD(X, z, 50, np.random.randn(int((deg+1)*(deg+2)/2)), 1000, 0.1, 100)
