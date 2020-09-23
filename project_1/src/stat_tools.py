
def R2(y_data, y_model):
    # Computes the confidence number
    return 1 - np.sum((y_data - y_model)**2) / np.sum((y_data - np.mean(y_data))**2)


def MSE(y_data, y_model):
    # Computes the mean squared error
    return np.sum((y_data - y_model)**2) / np.size(y_model)

def var_beta(y_data, X):
    """ Computes the covariance matrix (only diagonal elements)
    Args:
        y_data (Array): Data points.
        X (Array): Design matrix corresponding to y_data

    Returns:
        Array: Covariance Matrix (diagonal elements)
    """
    return np.sqrt(np.var(y_data) * np.linalg.inv(X.T @ X).diagonal())



def compute_mse_bias_variance(y_data, y_model):
    """ Computes MSE, mean (squared) bias and mean variance for a given set of y_data and y_model, where
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
    mse = np.mean(np.mean((y_data[:,np.newaxis] - y_model)**2,axis=1,keepdims=True))
    mean_squared_bias = np.mean((y_data[:,np.newaxis]-np.mean(y_model,axis=1,keepdims=True))**2)
    mean_variance = np.mean(np.var(y_model,axis=1,keepdims=True))

    return mse, mean_squared_bias, mean_variance

def bootstrap_selection(z, n_bootstraps):
    """ Performs n_bootstraps, returning a list of arrays where each array
    contains the selection indices of z for that particular bootstrap-iteration.
    Deprecated
    Args:
            z (array): The training data to bootstrap.
            n_bootstraps (int): Number of bootstrap-iterations

    Retunrs:
        bootstrap_indices (list): List of arrays, where each array contains the
            selection-indices for a particular bootstrap-iteration.

    """
    bootstrap_indices = []
    for i in range(n_bootstraps):
        shuffle = np.random.randint(0,len(z),len(z))
        bootstrap_indices.append(shuffle)
    return bootstrap_indices

def k_fold_selection_simplistic(z,k):
    """ Takes a vector z, retunrs two lists of k elements, each element
        being an array of indices for a permuted selection of z. The second
        list being the complementary set (i.e. excluding) of the test indices
        from the first list. This is the way too simple and studpid version.
        Deprecated.

        Args:
            z (1d-array): Vector to do a k-fold splitting on
            k (int): Nunber of folds


        Returns:
            test_indices (list): Each element in the list is a 1d-array containing
                                 the indices for the test-data for one of the k
                                 selections
            train_indices (list): Each element in the list is a 1d-array containt
                                  the indices for the training-data for one of the k
                                  selections

    """

    k_folds = k
    test_indices = []
    train_indices = []
    fold_number = np.random.randint(0,k_folds,len(z))
    for k in range(k_folds):
        test_index = np.where(fold_number == k)
        train_index = np.where(fold_number != k)
        test_indices.append(test_index)
        train_indices.append(test_index)
        # Commented below are examples for how to use the indices:
        # z_folded_test = z[test_index]
        # x_folded_test = X[test_index]
        # x_folded = X[test_index]
        # z_folded = z[train_index]
    return test_indices, train_indices



def k_fold_selection(z,k):
    """ Takes a vector z, retunrs two lists of k elements, each element
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
    elements_per_bin = int(len(z)/k_folds)
    permutations = np.random.permutation(np.arange(len(z)))
    for k in range(k_folds):

        # Create a mask which is True/False for respectively train/test
        # Moves along the permutation vector picking elements_per_bin as False
        # for each k. Essentially a fancy way to slice and exclude on the permutations
        test_mask = np.ones(len(z), bool)
        test_mask[k*elements_per_bin:(k+1)*elements_per_bin] = False
        if k == k_folds-1:
            test_mask[(k+1)*elements_per_bin:] = False
        test_indices.append(permutations[np.logical_not(test_mask)])
        train_indices.append(permutations[test_mask])

        # Commented below are examples for how to use the indices:
        # z_folded_test = z[permutations[np.logical_not(test_mask)]]
        # X_folded_test = X[permutations[np.logical_not(test_mask)]]
        # z_folded_train = z[permutations[test_mask]]
        # X_folded_train = X[permutations[test_mask]]

    return test_indices, train_indices


def bootstrap_all(X_train, X_test, z_train, z_test, n_bootstraps, lamb_lasso, lamb_ridge):
    """ Performs the bootstrapped bias variance analysis for OLS, Ridge and Lasso, given input
        training and test data, the number of bootstrap iterations and the lambda values for
        Ridge and Lasso.

        Returns MSE, mean squared bias and mean variance for Ridge, Lasso and OLS in that order.
    """

    z_boot_ols = np.zeros((len(z_test),n_bootstraps))
    z_boot_ridge = np.zeros((len(z_test),n_bootstraps))
    z_boot_lasso= np.zeros((len(z_test),n_bootstraps))
    for i in range(n_bootstraps):
        shuffle = np.random.randint(0,len(z_train),len(z_train))
        X_boot, z_boot = X_train[shuffle] , z_train[shuffle]
        betas_boot_ols = linear_regression.OLS_SVD_2D(X_boot, z_boot)
        betas_boot_ridge = linear_regression.Ridge_2D(X_boot, z_boot, lamb_ridge) #Ridge, given lambda
        clf_Lasso = skl.Lasso(alpha=lamb_lasso,fit_intercept=False).fit(X_boot,z_boot)
        z_boot_lasso[:,i] = clf_Lasso.predict(X_test) #Lasso, given lambda
        z_boot_ridge[:,i] = X_test @ betas_boot_ridge
        z_boot_ols[:,i] = X_test @ betas_boot_ols

    ridge_mse, ridge_bias, ridge_variance = stat_tools.compute_mse_bias_variance(z_test, z_boot_ridge)

    lasso_mse, lasso_bias, lasso_variance = stat_tools.compute_mse_bias_variance(z_test, z_boot_lasso)

    ols_mse, ols_bias, ols_variance = stat_tools.compute_mse_bias_variance(z_test, z_boot_ols)

    return ridge_mse, ridge_bias, ridge_variance, lasso_mse, lasso_bias, lasso_variance, ols_mse, ols_bias, ols_variance

def bootstrap_ridge_lasso(X_train, X_test, z_train, z_test, n_bootstraps, lamb_lasso, lamb_ridge):
    """ Performs the bootstrapped bias variance analysis for only Ridge and Lasso, given input
        training and test data, the number of bootstrap iterations and the lambda values for
        Ridge and Lasso. Intended for studying bias/variance dependency as a function of lambda-values.

        Returns MSE, mean squared bias and mean variance for Ridge and Lasso, in that order
    """

    z_boot_ridge = np.zeros((len(z_test),n_bootstraps))
    z_boot_lasso= np.zeros((len(z_test),n_bootstraps))
    for i in range(n_bootstraps):
        shuffle = np.random.randint(0,len(z_train),len(z_train))
        X_boot, z_boot = X_train[shuffle] , z_train[shuffle]
        betas_boot_ridge = linear_regression.Ridge_2D(X_boot, z_boot, lamb_ridge) #Ridge, given lambda
        clf_Lasso = skl.Lasso(alpha=lamb_lasso,fit_intercept=False).fit(X_boot,z_boot)
        z_boot_lasso[:,i] = clf_Lasso.predict(X_test) #Lasso, given lambda
        z_boot_ridge[:,i] = X_test @ betas_boot_ridge

    ridge_mse, ridge_bias, ridge_variance = stat_tools.compute_mse_bias_variance(z_test, z_boot_ridge)

    lasso_mse, lasso_bias, lasso_variance = stat_tools.compute_mse_bias_variance(z_test, z_boot_lasso)

    return ridge_mse, ridge_bias, ridge_variance, lasso_mse, lasso_bias, lasso_variance



def k_fold_cv_all(X,z,n_lambdas,lambdas,k_folds):
    """ Performs k-fold cross validation for Ridge, Lasso and OLS. The Lasso and Ridge
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
    test_list, train_list = stat_tools.k_fold_selection(z, k_folds)
    for i in range(n_lambdas):
        lamb = lambdas[i]
        for j in range(k_folds):
            test_ind_cv = test_list[j]
            train_ind_cv = train_list[j]
            X_train_cv = X[train_ind_cv]
            z_train_cv = z[train_ind_cv]
            X_test_cv = X[test_ind_cv]
            z_test_cv = z[test_ind_cv]
            clf_Lasso = skl.Lasso(alpha=lamb,fit_intercept=False).fit(X_train_cv,z_train_cv)
            z_lasso_test = clf_Lasso.predict(X_test_cv)
            ridge_betas = linear_regression.Ridge_2D(X_train_cv, z_train_cv, lamb)
            z_ridge_test = X_test_cv @ ridge_betas
            ridge_fold_score[i,j] = stat_tools.MSE(z_test_cv, z_ridge_test)
            lasso_fold_score[i,j] = stat_tools.MSE(z_test_cv, z_lasso_test)

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


def k_folds_cv_OLS_only(X,z,k_folds):
    """ As could be guessed, computes the k-fold cross-validation MSE for OLS, given
        input X, y as data; k_folds as number of folds. Returns the computed MSE.

    """


    ols_fold_score = np.zeros(k_folds)
    test_list, train_list = stat_tools.k_fold_selection(z, k_folds)
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

    return ols_cv_mse
