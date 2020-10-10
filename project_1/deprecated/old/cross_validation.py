import numpy as np
import pandas as pd

import stat_tools
import linear_regression


def cross_validation(X, z, k_folds, regression=linear_regression.OLS_SVD_2D):

    columns = [
        "MSE train",
        "MSE test",
        "R2 train",
        "R2 test"
        # "Variance train", "Variance test"
        # "Bias train", "Bias test"
    ]
    dat = pd.DataFrame(columns=columns, index=np.arange(k_folds))
    test_indices, train_indices = stat_tools.k_fold_selection(z, k=k_folds)

    for k in range(k_folds):
        # Training data
        X_train = X[train_indices[k], :]
        z_train = z[train_indices[k]]
        # Testing data
        X_test = X[test_indices[k], :]
        z_test = z[test_indices[k]]

        beta = regression(X_train, z_train)

        dat["MSE train"][k] = stat_tools.MSE(z_train, X_train @ beta)
        dat["R2 train"][k] = stat_tools.R2(z_train, X_train @ beta)

        dat["MSE test"][k] = stat_tools.MSE(z_test, X_test @ beta)
        dat["R2 test"][k] = stat_tools.R2(z_test, X_test @ beta)

    return dat
