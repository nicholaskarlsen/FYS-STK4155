import numpy as np
import linear_regression
import pandas as pd
import stat_tools
import sklearn.linear_model as skl


def bootstrap(X_train, X_test, z_train, z_test, bootstraps, regression):
    """
    Assumes regression(X, z), but can take different parameters like
    bootstrap(..., regression = lambda X, y : regression_method(X, y, lamb)):
    """

    N = z_train.size  # Number of data points in training set
    z_model_test = np.empty((bootstraps, z_test.size))  # Storage for the bootstrapped test data

    for i in range(bootstraps):
        # Generate N random indices from 0, N
        indices = np.random.randint(0, N, N)
        # Fetch out the bootstrap data sets
        X_boot = X_train[indices, :]
        z_bootstrap = z_train[indices]

        beta = regression(X_boot, z_bootstrap)
        z_model_test[i, :] = X_test @ beta

    MSE_test = stat_tools.MSE(z_test, z_model_test)
    bias2_test = stat_tools.mean_squared_bias(z_test, z_model_test)
    variance_test = stat_tools.mean_variance(z_test, z_model_test)

    return MSE_test, bias2_test, variance_test


# Special treatment due to differing interface
def bootstrap_lasso(X_train, X_test, z_train, z_test, bootstraps, lambd):
    N = z_train.size  # Number of data points in training set
    z_model_test = np.empty((bootstraps, z_test.size))  # Storage for the bootstrapped test data

    for i in range(bootstraps):
        # Generate N random indices from 0, N
        indices = np.random.randint(0, N, N)
        # Fetch out the bootstrap data sets
        X_boot = X_train[indices, :]
        z_boot = z_train[indices]

        clf_Lasso = skl.Lasso(alpha=lambd,fit_intercept=False).fit(X_boot, z_boot)
        z_model_test[i, :] = clf_Lasso.predict(X_test)

    MSE_test = stat_tools.MSE(z_test, z_model_test)
    bias2_test = stat_tools.mean_squared_bias(z_test, z_model_test)
    variance_test = stat_tools.mean_variance(z_test, z_model_test)

    return MSE_test, bias2_test, variance_test
