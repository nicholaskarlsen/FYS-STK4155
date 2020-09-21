import numpy as np
import linear_regression
import pandas as pd
from main import MSE, R2
import stat_tools

def bootstrap(X_train, X_test, z_train, z_test, bootstraps=100, regression=linear_regression.OLS_SVD_2D):

	columns = [
		"MSE train", "MSE test",
	#	"R2 train", "R2 test"
		"Bias train", "Bias test",
		"Variance train", "Variance test"
	]

	data = pd.DataFrame(columns = columns, index = np.arange(0, bootstraps))

	N = len(z_train)

	z_model_train = np.zeros([bootstraps, N])
	z_model_test = np.zeros([bootstraps, len(z_test)])

	for i in range(bootstraps):
		indices = np.random.randint(0, N, N)
		X_boot = X_train[indices, :]
		z_boot = z_train[indices]
		beta = regression(X_boot, z_boot)

		z_model_train[i, :] = X_boot @ beta
		z_model_test[i, :] = X_test @ beta

	#data["MSE train"][i] = MSE(z_boot, z_boot_tilde)
	#data["R2 train"][i] = R2(z_boot, z_boot_tilde)
	#data["MSE test"][i] = MSE(z_test, z_test_tilde)
	#data["R2 test"][i] = R2(z_test, z_test_tilde)

	data["MSE train"], data["Bias train"], data["Variance train"] \
		= stat_tools.compute_mse_bias_variance(z_train, z_model_train)

	data["MSE test"], data["Bias test"], data["Variance test"] \
		= stat_tools.compute_mse_bias_variance(z_test, z_model_test)

	print(len(data["MSE train"]))

	return data
