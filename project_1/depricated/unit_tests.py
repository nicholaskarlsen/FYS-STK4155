import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge

import linear_regression
from FrankeFunction import FrankeFunction


def OLS_unit_test(min_deg=0, max_deg=15, tol=1e-6):
    n = 100  # Number of data points
    # Prepare data set
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    z = FrankeFunction(x, y) + np.random.normal(0, 1, n) * 0.2
    degrees = np.arange(min_deg, max_deg + 1)
    for deg in degrees:
        # Set up design matrix
        X = linear_regression.design_matrix(x, y, 5)
        # Compute optimal parameters using our homegrown OLS
        beta = linear_regression.OLS(X=X, z=z)
        # Compute optimal parameters using sklearn
        skl_reg = LinearRegression(fit_intercept=False).fit(X, z)
        beta_skl = skl_reg.coef_

        for i in range(len(beta)):
            if abs(beta[i] - beta_skl[i]) < tol:
                pass
            else:
                print("Warning! mismatch with SKL in OLS_unit_test with tol = %.0e" % tol)
                print("Parameter no. %i for deg = %i" % (i, deg))
                print("-> (OUR) beta = %8.12f" % beta[i])
                print("-> (SKL) beta = %8.12f" % beta_skl[i])
    return


def OLS_SVD_unit_test(min_deg=0, max_deg=15, tol=1e-6):
    n = 100  # Number of data points
    # Prepare data set
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    z = FrankeFunction(x, y) + np.random.normal(0, 1, n) * 0.2
    degrees = np.arange(min_deg, max_deg + 1)
    for deg in degrees:
        # Set up design matrix
        X = linear_regression.design_matrix(x, y, 5)
        # Compute optimal parameters using our homegrown OLS
        beta = linear_regression.OLS_SVD(X=X, z=z)
        # Compute optimal parameters using sklearn
        skl_reg = LinearRegression(fit_intercept=False).fit(X, z)
        beta_skl = skl_reg.coef_

        for i in range(len(beta)):
            if abs(beta[i] - beta_skl[i]) < tol:
                pass
            else:
                print("Warning! mismatch with SKL in OLS_SVD_unit_test with tol = %.0e" % tol)
                print("Parameter no. %i for deg = %i" % (i, deg))
                print("-> (OUR) beta = %8.12f" % beta[i])
                print("-> (SKL) beta = %8.12f" % beta_skl[i])
    return


def Ridge_unit_test(min_deg=0, max_deg=15, tol=1e-10):
    n = 100  # Number of data points
    # Prepare data set
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    z = FrankeFunction(x, y) + np.random.normal(0, 1, n) * 0.2
    degrees = np.arange(min_deg, max_deg + 1)
    for deg in degrees:
        # Set up design matrix
        X = linear_regression.design_matrix(x, y, 5)
        for lamb in np.logspace(-3, 0, 30):
            # Compute optimal parameters using our homegrown Ridge regression
            beta = linear_regression.Ridge(X=X, z=z, lamb=lamb)
            # Compute optimal parameters using sklearn
            skl_reg = Ridge(alpha=lamb, fit_intercept=False).fit(X, z)
            beta_skl = skl_reg.coef_
            for i in range(len(beta)):
                if abs(beta[i] - beta_skl[i]) < tol:
                    pass
                else:
                    print("Warning! mismatch with SKL in Ridge_unit_test with tol = %.0e" % tol)
                    print("Parameter no. %i for deg = %i" % (i, deg))
                    print("-> (OUR) beta = %8.12f" % beta[i])
                    print("-> (SKL) beta = %8.12f" % beta_skl[i])
    return


if __name__ == "__main__":
    OLS_unit_test()
    OLS_SVD_unit_test()
    Ridge_unit_test()
