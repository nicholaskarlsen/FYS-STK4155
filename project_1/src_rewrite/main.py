import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

import stat_tools
import linear_regression
import bootstrap
import cross_validation
from sklearn.preprocessing import StandardScaler


def bootstrap_analysis(x, y, z, degrees, N_bootstraps, regression=linear_regression.OLS_SVD, lambd = None):

    scaler = StandardScaler()
    MSE_test = np.empty(degrees.size)
    bias2_test = np.empty(degrees.size)
    variance_test = np.empty(degrees.size)

    for i, deg in enumerate(degrees):
        X = linear_regression.design_matrix(x, y, deg)
        # Split data, but don't shuffle. OK since data is already randomly sampled!
        # Facilitates a direct comparison of the clean & Noisy data
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, shuffle=False)

        # Normalize data sets
        X_train = scaler.fit_transform(X_train)
        X_train[:, 0] = np.ones(X_train.shape[0])
        X_test = scaler.fit_transform(X_test)
        X_test[:, 0] = np.ones(X_test.shape[0])

        # Lasso requires special treatment due to difference between SKL's way of doing things
        # And our codes

        if regression != "LASSO":
            MSE_test[i], bias2_test[i], variance_test[i] = bootstrap.bootstrap(
                X_train, X_test, z_train, z_test, bootstraps=N_bootstraps, regression=regression
            )
        else:
            MSE_test[i], bias2_test[i], variance_test[i] = bootstrap.bootstrap_lasso(
                X_train, X_test, z_train, z_test, bootstraps=N_bootstraps, lambd=lambd
            )

    return MSE_test, bias2_test, variance_test




def cv_analysis(x, y, z, degrees, k, regression=linear_regression.OLS_SVD, lambd = None):

    scaler = StandardScaler()
    MSE_test = np.empty(degrees.size)

    for i, deg in enumerate(degrees):
        X = linear_regression.design_matrix(x, y, deg)
        # Normalize the design matrix
        X = scaler.fit_transform(X)
        X[:, 0] = np.ones(X.shape[0])

        if regression != "LASSO":
            MSE_test[i] = cross_validation.cross_validation(X, z, k_folds=k, regression=regression)
        else:
            MSE_test[i] = cross_validation.cross_validation_lasso(X, z, k_folds=k, lambd=lambd)


    return MSE_test
