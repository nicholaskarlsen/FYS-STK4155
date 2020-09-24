import numpy as np
import linear_regression
import pandas as pd
import stat_tools


def bootstrap(X_train, X_test, z_train, z_test, bootstraps, regression):
    """
        Assumes regression(X, z), but can take different parameters like
        bootstrap(..., regression = lambda X, y : regression_method(X, y, lamb)):
    """

    N = len(z_train)  # Number of data points in training set

    z_bootstrap = np.zeros([bootstraps, N])  # Storage for the bootstrapped data
    z_model_train = np.zeros([bootstraps, N])  # Storage for the bootstrapped model data
    z_model_test = np.zeros([bootstraps, len(z_test)])  # Storage for the bootstrapped test data

    for i in range(bootstraps):
        # Generate N random indices from 0, N
        indices = np.random.randint(0, N, N)
        # Fetch out the bootstrap data sets
        X_boot = X_train[indices, :]
        z_bootstrap[i, :] = z_train[indices]
        beta = regression(X_boot, z_bootstrap[i, :])

        z_model_train[i, :] = X_boot @ beta
        z_model_test[i, :] = X_test @ beta

    # Compute all of the statistics
    MSE_train = stat_tools.MSE(z_bootstrap, z_model_train)
    MSE_test = stat_tools.MSE(z_test, z_model_test)

    R2_train = stat_tools.R2(z_bootstrap, z_model_train)
    R2_test = stat_tools.R2(z_test, z_model_test)

    bias_train = stat_tools.mean_squared_bias(z_bootstrap, z_model_train)
    bias_test = stat_tools.mean_squared_bias(z_test, z_model_test)

    variance_train = stat_tools.mean_variance(z_bootstrap, z_model_train)
    variance_test = stat_tools.mean_variance(z_test, z_model_test)

    # Package results in a neat little package
    # In order to lessen the chance of missasignment
    data = {
        "MSE train": [MSE_train],
        "MSE test": [MSE_test],
        "R2 train": [R2_train],
        "R2 test": [R2_test],
        "Bias train": [bias_train],
        "Bias test": [bias_train],
        "Variance train": [variance_train],
        "Variance test": [variance_test],
    }

    return data
