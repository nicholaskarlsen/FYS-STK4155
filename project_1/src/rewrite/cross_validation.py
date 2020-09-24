import numpy as np
import stat_tools
import linear_regression


def cross_validation(X, z, k_folds, regression):
    # Initialize outgoing arrays
    MSE_train = np.zeros(k_folds)
    MSE_test = np.zeros(k_folds)
    R2_train = np.zeros(k_folds)
    R2_test = np.zeros(k_folds)

    test_indices, train_indices = stat_tools.k_fold_selection(z, k_folds=k_folds)

    N_test = len(test_indices[0])
    N_train = len(train_indices[0])

    for k in range(k_folds):
        # Training data
        X_train = X[train_indices[k], :]
        z_train = z[train_indices[k]]
        # Testing data
        X_test = X[test_indices[k], :]
        z_test = z[test_indices[k]]
        # Solve model
        beta = regression(X_train, z_train)

        # Compute statistics
        MSE_train[k] = stat_tools.MSE(z_train, X_train @ beta)
        MSE_test[k] = stat_tools.MSE(z_test, X_test @ beta)

        R2_train[k] = stat_tools.R2(z_train, X_train @ beta)
        R2_test[k] = stat_tools.R2(z_test, X_test @ beta)

    # Package results in a neat little package
    # In order to lessen the chance of missasignment
    data = {
        "MSE train": [np.mean(MSE_train)],
        "MSE test": [np.mean(MSE_test)],
        "R2 train": [np.mean(R2_train)],
        "R2 test": [np.mean(R2_test)],
    }

    return data
