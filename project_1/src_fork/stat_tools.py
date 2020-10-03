import numpy as np
import sklearn.linear_model as skl
import linear_regression


def R2(y_data, y_model):
    # Computes the confidence number
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)


def MSE(y_data, y_model):
    # Computes the mean squared error
    return np.sum((y_data - y_model) ** 2) / np.size(y_model)


def var_beta(X, sigma=1):
    """Computes the diagonal elements of the covariance matrix
    Args:
        y_data (Array): Data points.
        X (Array): Design matrix corresponding to y_data
    Returns:
        Array: Covariance Matrix (diagonal elements)
    """
    return np.sqrt(np.linalg.pinv(X.T @ X).diagonal()) * sigma**2
