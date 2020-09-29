import numpy as np
import stat_tools
import linear_regression
import sklearn.linear_model as skl


def cross_validation(X, z, k_folds, regression):
    # Initialize outgoing arrays
    MSE_test = np.zeros(k_folds)
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
        MSE_test[k] = stat_tools.MSE(z_test, X_test @ beta)


    # Package results in a neat little package
    # In order to lessen the chance of missasignment

    return np.mean(MSE_test)


def cross_validation_lasso(X, z, k_folds, lambd):
    # Initialize outgoing arrays
    MSE_test = np.zeros(k_folds)
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
        clf_Lasso = skl.Lasso(alpha=lambd,fit_intercept=False).fit(X_train, z_train)
        z_model_test = clf_Lasso.predict(X_test)

        # Compute statistics
        MSE_test[k] = stat_tools.MSE(z_test, z_model_test)


    # Package results in a neat little package
    # In order to lessen the chance of missasignment

    return np.mean(MSE_test)
