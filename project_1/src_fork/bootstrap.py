import numpy as np
import sklearn.linear_model as skl
import linear_regression

from stat_tools import *


def compute_mse_bias_variance(y_data, y_model):
    """Computes MSE, mean (squared) bias and mean variance for a given set of y_data and y_model, where
        each column of y_model comes from a particular realization of the model.
        The averages are first taken over the models, then over the data points.
        To be clear, error bias and variance are computed as the ensemble averages
        over the training set ensembles seperately for each test point. Only then
        are the means over the number of test points of those quantities computed and returned.
    Args:
        y_data (Array): the data-values for the test set.
        y_model (Array): the model values corresponding to the test values. 2d-
            array, where the second dimension corresponds to the number of training-
            iterations. E.g. the number of bootstraps.

    Returns:
        mean_squared_bias (float): the mean (squared) model bias for the given inputs
        mean_variance (float): the mean model variance for the given inputs
    """
    mse = np.mean(np.mean((y_data[:, np.newaxis] - y_model) ** 2, axis=1, keepdims=True))
    mean_squared_bias = np.mean((y_data[:, np.newaxis] - np.mean(y_model, axis=1, keepdims=True)) ** 2)
    mean_variance = np.mean(np.var(y_model, axis=1, keepdims=True))

    return mse, mean_squared_bias, mean_variance


def bootstrap(X_train, X_test, z_train, z_test, n_bootstraps, regression):
    """
    performs bootstrap for OLS & Ridge. Not lasso due to difference of interface.
    """
    N = z_train.size
    z_boot= np.empty((z_test.size, n_bootstraps))

    for i in range(n_bootstraps):
        shuffle = np.random.randint(0, N, N)
        X_boot, z_boot = X_train[shuffle], z_train[shuffle]
        betas = regression(X_boot, z_boot)
        z_model_test[:, i] = X_test @ beta

    mse, bias, var = compute_mse_bias_variance(z_test, z_boot)
    return


def bootstrap_lasso(X_train, X_test, z_train, z_test, n_bootstraps, lambd):
    """
    Performs bootstrap for LASSO.
    """
    N = z_train.size
    z_boot= np.empty((z_test.size, n_bootstraps))

    for i in range(n_bootstraps):
        shuffle = np.random.randint(0, N, N)
        X_boot, z_boot = X_train[shuffle], z_train[shuffle]
        clf_Lasso = skl.Lasso(alpha=lambd,fit_intercept=False).fit(X_boot, z_boot)
        z_model_test[:, i] = clf_Lasso.predict(X_test)

    mse, bias, var = compute_mse_bias_variance(z_test, z_boot)
    return


def bootstrap_all(X_train, X_test, z_train, z_test, n_bootstraps, lamb_lasso, lamb_ridge):
    """Performs the bootstrapped bias variance analysis for OLS, Ridge and Lasso, given input
    training and test data, the number of bootstrap iterations and the lambda values for
    Ridge and Lasso.

    Returns MSE, mean squared bias and mean variance for Ridge, Lasso and OLS in that order.
    """

    z_boot_ols = np.zeros((len(z_test), n_bootstraps))
    z_boot_ridge = np.zeros((len(z_test), n_bootstraps))
    z_boot_lasso = np.zeros((len(z_test), n_bootstraps))

    for i in range(n_bootstraps):
        shuffle = np.random.randint(0, len(z_train), len(z_train))
        X_boot, z_boot = X_train[shuffle], z_train[shuffle]
        betas_boot_ols = linear_regression.OLS_SVD_2D(X_boot, z_boot)
        # Ridge, given lambda
        betas_boot_ridge = linear_regression.Ridge_2D(X_boot, z_boot, lamb_ridge)
        clf_Lasso = skl.Lasso(alpha=lamb_lasso, fit_intercept=False).fit(X_boot, z_boot)
        z_boot_lasso[:, i] = clf_Lasso.predict(X_test)  # Lasso, given lambda
        z_boot_ridge[:, i] = X_test @ betas_boot_ridge
        z_boot_ols[:, i] = X_test @ betas_boot_ols

    ridge_mse, ridge_bias, ridge_variance = compute_mse_bias_variance(z_test, z_boot_ridge)
    lasso_mse, lasso_bias, lasso_variance = compute_mse_bias_variance(z_test, z_boot_lasso)
    ols_mse, ols_bias, ols_variance = compute_mse_bias_variance(z_test, z_boot_ols)

    return (
        ridge_mse,
        ridge_bias,
        ridge_variance,
        lasso_mse,
        lasso_bias,
        lasso_variance,
        ols_mse,
        ols_bias,
        ols_variance,
    )


def bootstrap_ridge_lasso(X_train, X_test, z_train, z_test, n_bootstraps, lamb_lasso, lamb_ridge):
    """Performs the bootstrapped bias variance analysis for only Ridge and Lasso, given input
    training and test data, the number of bootstrap iterations and the lambda values for
    Ridge and Lasso. Intended for studying bias/variance dependency as a function of lambda-values.

    Returns MSE, mean squared bias and mean variance for Ridge and Lasso, in that order
    """

    z_boot_ridge = np.zeros((len(z_test), n_bootstraps))
    z_boot_lasso = np.zeros((len(z_test), n_bootstraps))
    for i in range(n_bootstraps):
        shuffle = np.random.randint(0, len(z_train), len(z_train))
        X_boot, z_boot = X_train[shuffle], z_train[shuffle]
        betas_boot_ridge = linear_regression.Ridge_2D(X_boot, z_boot, lamb_ridge)  # Ridge, given lambda
        clf_Lasso = skl.Lasso(alpha=lamb_lasso, fit_intercept=False).fit(X_boot, z_boot)
        z_boot_lasso[:, i] = clf_Lasso.predict(X_test)  # Lasso, given lambda
        z_boot_ridge[:, i] = X_test @ betas_boot_ridge

    ridge_mse, ridge_bias, ridge_variance = compute_mse_bias_variance(z_test, z_boot_ridge)
    lasso_mse, lasso_bias, lasso_variance = compute_mse_bias_variance(z_test, z_boot_lasso)

    return ridge_mse, ridge_bias, ridge_variance, lasso_mse, lasso_bias, lasso_variance
