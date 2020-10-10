import numpy as np
import linear_regression
import pandas as pd
from main import MSE, R2
import stat_tools


def bootstrap(
    X_train, X_test, z_train, z_test, bootstraps=100, regression=linear_regression.OLS_SVD_2D
):

    N = len(z_train)

    # z_model_boot = np.zeros([bootstraps, N])
    # z_model_test = np.zeros([bootstraps, len(z_test)])

    mse_test = np.zeros(bootstraps)
    mse_train = np.zeros(bootstraps)

    for i in range(bootstraps):
        indices = np.random.randint(0, N, N)
        X_boot = X_train[indices, :]
        z_boot = z_train[indices]
        beta = regression(X_boot, z_boot)

        z_model_train = X_boot @ beta
        z_model_test = X_test @ beta

        mse_train[i] = stat_tools.MSE(z_boot, z_model_train)
        mse_test[i] = stat_tools.MSE(z_test, z_model_test)

    data = {
        "MSE train": [np.mean(mse_train)],  # [MSE(z_train, z_model_train)],
        "MSE test": [np.mean(mse_test)],  # [MSE(z_test, z_model_test)],
        "R2 train": [0],
        "R2 test": [0],
        "Bias train": [0],
        "Bias test": [0],
        "Variance train": [0],
        "Variance test": [0],
    }

    """
	data["MSE train"], data["Bias train"], data["Variance train"] \
		= stat_tools.compute_mse_bias_variance(z_train, z_model_train)

	data["MSE test"], data["Bias test"], data["Variance test"] \
		= stat_tools.compute_mse_bias_variance(z_test, z_model_test)

	print(len(data["MSE train"]))
	"""

    return data
