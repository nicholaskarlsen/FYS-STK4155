import numpy as np
import matplotlib.pyplot as plt

import stat_tools
import linear_regression
import bootstrap
import cross_validation


def part_1(x, y, z, degrees):

    mse = pd.DataFrame(columns=["train", "test"], index=degrees)
    r2 = pd.DataFrame(columns=["train", "test"], index=degrees)

    # var_b = pd.DataFrame(indices=degrees)

    for i, deg in enumerate(degrees):
        X = design_matrix(x, y, deg)
        # Split data, but don't shuffle. OK since data is already randomly sampled!
        # Fasilitates a direct comparrison of the clean & Noisy data
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, shuffle=False)
        # Normalize data sets
        X_train = scaler.fit_transform(X_train)
        X_train[:, 0] = np.ones(X_train.shape[0])
        X_test = scaler.fit_transform(X_test)
        X_test[:, 0] = np.ones(X_test.shape[0])

        beta = OLS_SVD(X_train, z_train)

        mse["train"][i] = MSE(z_train, X_train @ beta)
        mse["test"][i] = MSE(z_test, X_test @ beta)

        r2["train"][i] = R2(z_train, X_train @ beta)
        r2["test"][i] = R2(z_test, X_test @ beta)

        # var_b.append([deg, var_beta(z_train, X_train)])

    return mse, r2, 0  # , var_b
