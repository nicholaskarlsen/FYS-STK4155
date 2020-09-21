import numpy as np

def R2(y_data, y_model):
    # Computes the confidence number
    return 1 - np.sum((y_data - y_model)**2) / np.sum((y_data - np.mean(y_data))**2)


def MSE(y_data, y_model):
    # Computes the mean squared error
    return np.sum((y_data - y_model)**2) / np.size(y_model)

def var_beta(y_data, X):
    """ Computes the covariance matrix
    Args:
        y_data (Array): Data points.
        X (Array): Design matrix corresponding to y_data

    Returns:
        Array: Covariance Matrix
    """
    return np.sqrt(np.var(y_data) * np.linalg.inv(X.T @ X).diagonal())



def compute_mse_bias_variance(y_data, y_model):
    """ Computes MSE, bias and variance for a given set of y_data and y_model, where
        each column of y_model comes from a particular realization of the model.
        The averages are first taken over the models, then over the data points.
    Args:
        y_data (Array): the data-values for the test set.
        y_model (Array): the model values corresponding to the test values. 2d-
            array, where the second dimension corresponds to the number of training-
            iterations. E.g. the number of bootstraps.

    Returns:
        bias (float): the bias for the given inputs
        variance (float): the variance for the given inputs
    """
    mse = np.mean(np.mean((y_data - y_model)**2,axis=1,keepdims=True))
    bias = np.mean((y_data-np.mean(y_model,axis=1,keepdims=True))**2)
    variance = np.mean(np.var(y_model,axis=1,keepdims=True))

    return mse, bias, variance

def bootstrap_selection(z, n_bootstraps):
    """ Performs n_bootstraps, returning a list of arrays where each array
    contains the selection indices of z for that particular bootstrap-iteration.

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
