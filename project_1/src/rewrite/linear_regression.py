import numpy as np
import numba


@numba.jit(nopython=True)
def design_matrix(x, y, n):
    """Constructs a design matrix for 2D input data i.e f: (x, y) -> z where the target function
    is an n-th degree polynomial.

    Args:
        x (Array): x data points, i.e [x0,x1,...,xn]
        y (Array): y data points, i.e [y0,y1,...,yn]
        n (Int): Degree of polynomial to model

    Returns:
        [Array] : Design matrix yielding the least-squares solution
    """
    N = x.size  # Number of data points
    d = int((n + 1) * (n + 2) / 2.0)  # d.o.f for n-th degree polynomial
    X = np.empty((N, d))  # Design matrix

    X[:, 0] = 1

    for i in range(1, n + 1):
        q = int(i * (i + 1) / 2)  # Number of terms with degree < i
        for k in range(i + 1):
            X[:, q + k] = x ** (i - k) * y ** k

    return X


def OLS(X, z):
    """Computes the ordinary least squares solution of X -> (z) where X is the design
        matrix for an p-th degree polynomial fitting.
    Args:
        X (Array): Design matrix from design_matrix_2D
        z (Array): z data points, i.e [z0,z1,...,zn]

    Returns:
        beta (Array): The beta vector
    """
    beta = np.linalg.inv(X.T @ X) @ X.T @ z

    return beta


def OLS_SVD(X, z, use_np_pinv=True):
    """Computes the ordinary least squares solution of X -> (z) where X is the design
        matrix for an p-th degree polynomial fitting, using the SVD-inversion.
    Args:
        X (Array): Design matrix from design_matrix_2D
        z (Array): z data points, i.e [z0,z1,...,zn]
        use_np_pinv (bool): Set to True in order to use np.linalg.pinv instead.

    Returns:
        beta (Array): The beta vector
    """
    if use_np_pinv:
        beta = np.linalg.pinv(X) @ z

    else:

        U, s, V = np.linalg.svd(X)
        # Simple limit for detecting zero-valued singular values.
        tolerance = s[0] * 1e-14
        # Setting zero-values to actual zero
        reciprocal_s = np.where(s > tolerance, 1 / s, 0)
        # instead of machine-epsilon
        # Could just as well have used
        # np.linalg.pinv(X), they are
        # more clever with implementing
        # the tolerance.
        D = np.eye(len(U), len(V)) * reciprocal_s

        pseudo_inv = V.T @ D.T @ U.T
        beta = pseudo_inv @ z

    return beta


def Ridge(X, z, lamb):
    """Computes the ordinary least squares solution of X -> (z) where X is the design
        matrix for an n-th degree polynomial fitting.
    Args:
        X (Array): Design matrix from design_matrix_2D
        z (Array): z data points, i.e [z0,z1,...,zn]
        lamb (float): the hyper-parameter lambda.
    Returns:
        beta (Array): The beta vector
    """
    p_feat = len(X[0, :])  # number of columns/features in X
    I = np.eye(p_feat, p_feat)
    beta = np.linalg.inv(X.T @ X + lamb * I) @ X.T @ z

    return beta
