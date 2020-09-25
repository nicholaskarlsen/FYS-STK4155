import numpy as np
import sklearn.linear_model as skl


def R2(y_data, y_model):
    # Computes the confidence number
    return np.mean(
        1
        - np.sum((y_data - y_model) ** 2, axis=0, keepdims=True)
        / np.sum((y_data - np.mean(y_data, axis=0, keepdims=True)) ** 2, axis=0)
    )


def MSE(y_data, y_model):
    # Computes the mean squared error (Works for both "normal" situations & bootstrap!)
    return np.mean(np.mean((y_data - y_model) ** 2, axis=0))


def mean_variance(y_data, y_model):
    # Computes the variance of a particular data point in bootstrap
    return np.mean(np.var(y_model, axis=0))


def mean_squared_bias(y_data, y_model):
    return np.mean((y_data - np.mean(y_model, axis=0, keepdims=True)) ** 2)


def var_beta(y_data, X):
    """ Computes the covariance matrix (only diagonal elements)
    Args:
        y_data (Array): Data points.
        X (Array): Design matrix corresponding to y_data

    Returns:
        Array: Covariance Matrix (diagonal elements)
    """
    return np.var(y_data) * np.linalg.inv(X.T @ X).diagonal()


def k_fold_selection(z, k_folds):
    """Takes a vector z, retunrs two lists of k elements, each element
    being an array of indices for a permuted selection of z. The second
    list being the complementary set (i.e. excluding) of the test indices
    from the first list.

    Example of how to use the indices:
    z_folded_test = z[permutations[np.logical_not(test_mask)]]
    X_folded_test = X[permutations[np.logical_not(test_mask)]]
    z_folded_train = z[permutations[test_mask]]
    X_folded_train = X[permutations[test_mask]]

    Args:
        z (1d-array): Vector to do a k-fold splitting on
        k (int): Nunber of folds
    Returns:
        test_indices (list): Each element in the list is a 1d-array containing
                             the indices for the test-data for one of the k selections
        train_indices (list): Each element in the list is a 1d-array containt
                              the indices for the training-data for one of the k selections
    """
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
