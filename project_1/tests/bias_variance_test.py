import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

import sys
sys.path.insert(0,"../src/")

import linear_regression
import utils
import stat_tools
import bootstrap


def design_matrix_1D(x, degree):

    X = np.ones((len(x), degree + 1))
    for i in range(degree):
        X[:, i + 1] = x ** (i + 1)
    return X


def own_bias_variance(seed):

    # Seems to be identical with the lecture notes code. up to degree 25.
    # The deviations from that point on, though, are not of completely obvious origin.
    # Aha, using the same pipeline for the model gives the exact same bias-variance-curves.
    # Must be something where sklearn are clever with numerical errors, while we are not.
    # This does at least show that our bias and variance computations are CORRECT!!!!
    # Note, our compute_mse_bias_variance assumes the first argument to be a flat array.

    np.random.seed(seed)

    n = 400
    n_bootstraps = 100
    max_degree = 30
    n_boostraps = 100

    # Make data set.
    x = np.linspace(-3, 3, n).reshape(-1, 1)
    y = np.exp(-(x ** 2)) + 1.5 * np.exp(-((x - 2) ** 2)) + np.random.normal(0, 0.1, x.shape)
    z = y
    # y = np.zeros(n)
    x_train, x_test, z_train, z_test = train_test_split(x, z, test_size=0.2)

    # x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size = 0.2)

    # Quantities of interest:
    mse_ols_test = np.zeros(max_degree)
    mse_ols_train = np.zeros(max_degree)
    ols_cv_mse = np.zeros(max_degree)

    ols_boot_mse = np.zeros(max_degree)
    ols_boot_bias = np.zeros(max_degree)
    ols_boot_variance = np.zeros(max_degree)

    # Actual computations
    for degree in range(max_degree):
        # X = linear_regression.design_matrix_2D(x,y,degree)
        # X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2)
        # X = linear_regression.design_matrix_2D(x,y,degree)
        # X_train = linear_regression.design_matrix_2D(x_train,y_train,degree)
        # X_test = linear_regression.design_matrix_2D(x_test,y_test,degree)
        # Scaling and feeding to CV.
        # X = design_matrix_1D(x,degree)
        # X_train = design_matrix_1D(x_train,degree)
        # X_test = design_matrix_1D(x_test,degree)
        # scaler = StandardScaler()
        # scaler.fit(X)
        # X = scaler.transform(X)
        # X[:,0] = 1
        #
        # X_train_scaled = X_train
        # X_test_scaled = X_test

        # Scaling and feeding to bootstrap and OLS
        # scaler_boot = StandardScaler()
        # scaler_boot.fit(X_train)
        # X_train_scaled = scaler_boot.transform(X_train)
        # X_test_scaled = scaler_boot.transform(X_test)
        # X_train_scaled[:,0] = 1
        # X_test_scaled[:,0] = 1

        # OLS, get MSE for test and train set.

        model = make_pipeline(
            PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False)
        )
        y_pred = np.empty((z_test.shape[0], n_boostraps))
        for i in range(n_boostraps):
            x_, y_ = resample(x_train, z_train)
            y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()

        (
            ols_boot_mse[degree],
            ols_boot_bias[degree],
            ols_boot_variance[degree],
        ) = bootstrap.compute_mse_bias_variance(z_test.flatten(), y_pred)
    print(ols_boot_mse)
    plt.plot(ols_boot_mse, label="Error")
    plt.plot(ols_boot_bias, label="Bias_squared")
    plt.plot(ols_boot_variance, label="variance")
    plt.legend()
    # plt.show()

    return


def lecture_note_bias_variance(seed):

    np.random.seed(seed)

    n = 400
    n_boostraps = 100
    maxdegree = 30

    # Make data set.
    x = np.linspace(-3, 3, n).reshape(-1, 1)
    y = np.exp(-(x ** 2)) + 1.5 * np.exp(-((x - 2) ** 2)) + np.random.normal(0, 0.1, x.shape)
    error = np.zeros(maxdegree)
    bias = np.zeros(maxdegree)
    variance = np.zeros(maxdegree)
    polydegree = np.zeros(maxdegree)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    for degree in range(maxdegree):
        model = make_pipeline(
            PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False)
        )
        y_pred = np.empty((y_test.shape[0], n_boostraps))
        for i in range(n_boostraps):
            x_, y_ = resample(x_train, y_train)
            y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()

        polydegree[degree] = degree
        error[degree] = np.mean(np.mean((y_test - y_pred) ** 2, axis=1, keepdims=True))
        bias[degree] = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True)) ** 2)
        variance[degree] = np.mean(np.var(y_pred, axis=1, keepdims=True))
        print("Polynomial degree:", degree)
        print("Error:", error[degree])
        print("Bias^2:", bias[degree])
        print("Var:", variance[degree])
        print(
            "{} >= {} + {} = {}".format(
                error[degree], bias[degree], variance[degree], bias[degree] + variance[degree]
            )
        )

    plt.plot(polydegree, error, label="Error")
    plt.plot(polydegree, bias, label="bias")
    plt.plot(polydegree, variance, label="Variance")
    plt.legend()
    # plt.show()

    return


if __name__ == "__main__":
    plt.subplot(121)
    plt.title("Our implementation")
    own_bias_variance(2018)
    plt.subplot(122)
    plt.title("Lecture note implementation")
    lecture_note_bias_variance(2018)
    plt.show()
