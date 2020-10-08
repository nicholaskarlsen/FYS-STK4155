import numpy as np
import sklearn.linear_model as skl
import linear_regression

import stat_tools


def k_fold_selection(z, k):
    """
    Takes a vector z, retunrs two lists of k elements, each element
    being an array of indices for a permuted selection of z. The second
    list being the complementary set (i.e. excluding) of the test indices
    from the first list.

    Args:
        z (1d-array): Vector to do a k-fold splitting on
        k (int): Nunber of folds

    Returns:
        test_indices (list): Each element in the list is a 1d-array containing
                             the indices for the test-data for one of the k selections
        train_indices (list): Each element in the list is a 1d-array containt
                              the indices for the training-data for one of the k selections
    """
    k_folds = k
    test_indices = []
    train_indices = []
    elements_per_bin = int(len(z) / k_folds)
    permutations = np.random.permutation(np.arange(len(z)))
    for k in range(k_folds):
        # Create a mask which is True/False for respectively train/test
        # Moves along the permutation vector picking elements_per_bin as False
        # for each k. Essentially a fancy way to slice and exclude on the permutations
        test_mask = np.ones(len(z), bool)
        test_mask[k * elements_per_bin : (k + 1) * elements_per_bin] = False

        if k == k_folds - 1:
            test_mask[(k + 1) * elements_per_bin :] = False

        test_indices.append(permutations[np.logical_not(test_mask)])
        train_indices.append(permutations[test_mask])

    return test_indices, train_indices


def crossvalidation(X, z, k_folds, regression):
    """
    Performs k-fold regression for a linear regression model and computes the resulting MSE

    Note: Hypoer parameters (like the lambdas) can be inserted into regression via lambda functions.
    i.e lambda X, z : ridge(X, z, lambd = 1e-3)

    Note 2: LASSO regression is performed as a separate, specialized method due to sci-kit learn
    adhering to a different interface than our codes.

    Args:
        X (array) : Design matrix
        z (array) : Response values
        k_folds (Int) : Number of "folds"/splits to split the data into.
        regression (Function) : Regression method to use. Assumes form: regression(X, z)
            See above for how to insert additional parameters.
    """
    cv_mse = np.zeros(k_folds)  # Storage for the individual MSE scores
    test_list, train_list = k_fold_selection(z, k_folds)  # the k-fold split indices

    for i in range(k_folds):
        test_ind_cv = test_list[j]
        train_ind_cv = train_list[j]
        # Separate out the training and test data for the current split
        X_train_cv = X[train_ind_cv]
        z_train_cv = z[train_ind_cv]
        X_test_cv = X[test_ind_cv]
        z_test_cv = z[test_ind_cv]
        # Perform the regression in the usual way & compute resulting MSE
        cv_betas = regression(X_train_cv, z_train_cv)
        z_model = X_test_cv @ cv_betas
        cv_mse[i] = stat_tools.MSE(z_test_cv, z_model)

    ols_cv_mse = np.mean(ols_fold_score)

    return ols_cv_mse


def crossvalidation_OLS(X, z, k_folds):
    # Wrapper function, see crossvalidation
    return crossvalidation(X, z, k_folds, regression=linear_regression.OLS_SVD_2D(X, z))


def crossvalidation_Ridge(X, z, k_folds, lambd):
    # Wrapper function, see crossvalidation
    return crossvalidation(X, z, k_folds, regression=lambda X, z: linear_regression.Ridge_2D(X, z, lambd))


def crossvalidation_LASSO(X, z, k_folds, lambd):
    # LASSO requires special treatment due to different interface. Functionally the same as crossvalidation()
    cv_mse = np.zeros(k_folds)  # Storage for the individual MSE scores
    test_list, train_list = k_fold_selection(z, k_folds)  # the k-fold split indices

    for i in range(k_folds):
        test_ind_cv = test_list[j]
        train_ind_cv = train_list[j]
        # Separate out the training and test data for the current split
        X_train_cv = X[train_ind_cv]
        z_train_cv = z[train_ind_cv]
        X_test_cv = X[test_ind_cv]
        z_test_cv = z[test_ind_cv]
        # Perform the regression in the usual way
        clf_Lasso = skl.Lasso(alpha=lamb, fit_intercept=False).fit(X_train_cv, z_train_cv)
        z_lasso_test = clf_Lasso.predict(X_test_cv)
        lasso_fold_score[i] = stat_tools.MSE(z_test_cv, z_lasso_test)

    ols_cv_mse = np.mean(ols_fold_score)

    return ols_cv_mse


###########################################################
### Wrapper functions for generating statistics en masse ##
###########################################################

# PS: These functions adhere to a different naming scheme to retain backward compatability with existing codes


def k_folds_cv_OLS_only(X, z, k_folds):
    return crossvalidation_OLS(X, z, k_folds)


def k_fold_cv_all(X, z, n_lambdas, lambdas, k_folds):
    """
    Performs k-fold cross validation for Ridge, Lasso and OLS. The Lasso and Ridge
    MSE-values are computed for a number of n_lambdas, with the lambda values given
    by the lambdas array. OLS is done only once for each of the k_folds folds.

    Args:
        X (array): Design matrix
        z (array): Data-values/response-values/whatever-they-are-called-in-your-field-values
        n_lambdas (int): number of lambda values to use for Lasso and Ridge.
        lambdas (array): The actual lambda-values to try.
        k_folds (int): The number of folds.

    Return:
        lasso_cv_mse (array): array containing the computed MSE for each lambda in Lasso
        ridge_cv_mse (array): array containing the computed MSE for each lambda in Ridge
        ols_cv_mse (float): computed MSE for OLS.
    """

    ridge_fold_score = np.zeros((n_lambdas, k_folds))
    lasso_fold_score = np.zeros((n_lambdas, k_folds))
    test_list, train_list = k_fold_selection(z, k_folds)
    for i in range(n_lambdas):
        lamb = lambdas[i]
        for j in range(k_folds):
            test_ind_cv = test_list[j]
            train_ind_cv = train_list[j]
            X_train_cv = X[train_ind_cv]
            z_train_cv = z[train_ind_cv]
            X_test_cv = X[test_ind_cv]
            z_test_cv = z[test_ind_cv]
            clf_Lasso = skl.Lasso(alpha=lamb, fit_intercept=False).fit(X_train_cv, z_train_cv)
            z_lasso_test = clf_Lasso.predict(X_test_cv)
            ridge_betas = linear_regression.Ridge_2D(X_train_cv, z_train_cv, lamb)
            z_ridge_test = X_test_cv @ ridge_betas
            ridge_fold_score[i, j] = stat_tools.MSE(z_test_cv, z_ridge_test)
            lasso_fold_score[i, j] = stat_tools.MSE(z_test_cv, z_lasso_test)

    lasso_cv_mse = np.mean(lasso_fold_score, axis=1)
    ridge_cv_mse = np.mean(ridge_fold_score, axis=1)

    # Get ols_mse for cv.
    ols_fold_score = np.zeros(k_folds)
    for i in range(k_folds):
        test_ind_cv = test_list[j]
        train_ind_cv = train_list[j]
        X_train_cv = X[train_ind_cv]
        z_train_cv = z[train_ind_cv]
        X_test_cv = X[test_ind_cv]
        z_test_cv = z[test_ind_cv]
        ols_cv_betas = linear_regression.OLS_SVD_2D(X_train_cv, z_train_cv)
        z_ols_test = X_test_cv @ ols_cv_betas
        ols_fold_score[i] = stat_tools.MSE(z_test_cv, z_ols_test)

    ols_cv_mse = np.mean(ols_fold_score)

    return lasso_cv_mse, ridge_cv_mse, ols_cv_mse
