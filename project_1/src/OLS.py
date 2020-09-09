import numpy as np
import linear_regression


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
    X = linear_regression.design_matrix_2D(x, y, n)
    beta = np.linalg.inv(X.T @ X) @ X.T @ z

    return beta
