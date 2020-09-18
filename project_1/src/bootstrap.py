import numpy as np
import linear_regression
from main import MSE, R2

def bootstrap(X_train, X_test, z_train, z_test, bootstraps=100, regression=linear_regression.OLS_SVD_2D):
	MSE_train = np.zeros(bootstraps)
	R2_train = np.zeros(bootstraps)
	MSE_test = np.zeros(bootstraps)
	R2_test = np.zeros(bootstraps)


	N = len(z_train)

	for i in range(bootstraps):
		indices = np.random.randint(0, N, N)
		X_boot = X_train[indices, :]
		z_boot = z_train[indices]
		beta = regression(X_boot, z_boot)

		z_boot_tilde = X_boot @ beta
		z_test_tilde = X_test @ beta
		MSE_train[i] = MSE(z_boot, z_boot_tilde)
		R2_train[i] = R2(z_boot, z_boot_tilde)
		MSE_test[i] = MSE(z_test, z_test_tilde)
		R2_test[i] = R2(z_test, z_test_tilde)

	MSE_train = np.mean(MSE_train)
	MSE_test = np.mean(MSE_test)
	R2_train = np.mean(R2_train)
	R2_test = np.mean(R2_test)

	return MSE_train, MSE_test, R2_train, R2_test