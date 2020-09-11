import numpy as np


def design_matrix_2D(x, y, n):
    """Constructs a design matrix for 2D input data i.e f: (x, y) -> z where the target function
    is an n-th degree polynomial.

    Args:
        x (Array): x data points, i.e [x0,x1,...,xn]
        y (Array): y data points, i.e [y0,y1,...,yn]
        n (Int): Degree of polynomial to model

    Returns:
        [Array] : Design matrix yielding the least-squares solution
    """
    N = len(x)                          # Number of data points
    d = int((n + 1) * (n + 2) / 2.0)    # d.o.f for n-th degree polynomial
    X = np.zeros([N, d])                # Design matrix

    X[:, 0] = 1

    for i in range(1, n + 1):
        q = int(i * (i + 1) / 2)       # Number of terms with degree < i
        for k in range(i + 1):
            X[:, q + k] = x ** (i - k) * y ** k

            print(q+k, ":", "x^%i" %(i-k), "* y^%i" % k)

    return X


def evaluate_poly_2D(x, y, beta, n):
    """Evaluates a polynomial constructed by OLS_2D at (x,y) given an array containing betas
    Args:
        x (float: x-coordinate to evaluate polynomial (Can be array)
        y (float): y-coordinate to evaluate polynomial (Can be array)
        betas (Array): Free parameters in polynomial.
        n (int): degree of polynomial

    Returns:
        Float (or Array, depending on input): Polynomial evaluated at (x, y)
    """

    z = np.zeros(x.shape)
    z += beta[0]
    for i in range(1, n + 1):
        q = int(i * (i + 1) / 2)        # Number of terms with degree < i
        for k in range(i + 1):
            z += beta[q + k]  * x ** (i - k) * y**k

    return z


def OLS_2D(x, y, z, n=4):
    """Computes the ordinary least squares solution of (x, y) -> (z) unto an n-th degree polynomial
    Args:
        x (Array): x data points, i.e [x0,x1,...,xn]
        y (Array): y data points, i.e [y0,y1,...,yn]
        z (Array): z data points, i.e [z0,z1,...,zn]
        n (Int): Degree of polynomial to model

    Returns:
        [type]: [description]
    """
    X = design_matrix_2D(x, y, n)
    beta = np.linalg.inv(X.T @ X) @ X.T @ z

    return beta
