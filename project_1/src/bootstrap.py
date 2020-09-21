import numpy as np
import linear_regression
import pandas as pd
from main import MSE, R2

def bootstrap(X_train, X_test, z_train, z_test, bootstraps=100, regression=linear_regression.OLS_SVD_2D):

	columns = [
		"MSE train", "MSE test",
		"R2 train", "R2 test"
		#"Variance train", "Variance test"
		#"Bias train", "Bias test"
	]

	data = pd.DataFrame(columns = columns, index = np.arange(0, bootstraps))

	N = len(z_train)

	for i in range(bootstraps):
		indices = np.random.randint(0, N, N)
		X_boot = X_train[indices, :]
		z_boot = z_train[indices]
		beta = regression(X_boot, z_boot)

		z_boot_tilde = X_boot @ beta
		z_test_tilde = X_test @ beta

		data["MSE train"][i] = MSE(z_boot, z_boot_tilde)
		data["R2 train"][i] = R2(z_boot, z_boot_tilde)
		data["MSE test"][i] = MSE(z_test, z_test_tilde)
		data["R2 test"][i] = R2(z_test, z_test_tilde)


	return data
