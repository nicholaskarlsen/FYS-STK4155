import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0,"../src_fork/")

import linear_regression
import utils
import stat_tools
import crossvalidation
import bootstrap

utils.plot_settings()  # LaTeX fonts in Plots!


def test_function(x, y):
    return x ** 2 + x + 5


n = 100
noise_scale = 0.2
x = np.random.uniform(0, 1, n)
y = np.random.uniform(0, 1, n)
z = test_function(x, y)
# Adding standard normal noise:
z = z + noise_scale * np.random.normal(0, 1, len(z))
max_degree = 15
n_lambdas = 60
n_bootstraps = 100
k_folds = 5
lambdas = np.logspace(-12, 1, n_lambdas)
subset_lambdas = lambdas[::5]

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)

#   Centering the response
z_intercept = np.mean(z)
z = z - z_intercept

#   Centering the response
z_train_intercept = np.mean(z_train)
z_train = z_train - z_train_intercept
z_test = z_test - z_train_intercept

########### Setup of problem is completed above.

# Quantities of interest:
mse_ols_test = np.zeros(max_degree)
mse_ols_train = np.zeros(max_degree)
ols_cv_mse = np.zeros(max_degree)

ols_boot_mse = np.zeros(max_degree)
ols_boot_bias = np.zeros(max_degree)
ols_boot_variance = np.zeros(max_degree)

best_ridge_lambda = np.zeros(max_degree)
best_ridge_mse = np.zeros(max_degree)
ridge_best_lambda_boot_mse = np.zeros(max_degree)
ridge_best_lambda_boot_bias = np.zeros(max_degree)
ridge_best_lambda_boot_variance = np.zeros(max_degree)

best_lasso_lambda = np.zeros(max_degree)
best_lasso_mse = np.zeros(max_degree)
lasso_best_lambda_boot_mse = np.zeros(max_degree)
lasso_best_lambda_boot_bias = np.zeros(max_degree)
lasso_best_lambda_boot_variance = np.zeros(max_degree)

ridge_lamb_deg_mse = np.zeros((max_degree, n_lambdas))
lasso_lamb_deg_mse = np.zeros((max_degree, n_lambdas))

ridge_subset_lambda_boot_mse = np.zeros((max_degree, len(subset_lambdas)))
ridge_subset_lambda_boot_bias = np.zeros((max_degree, len(subset_lambdas)))
ridge_subset_lambda_boot_variance = np.zeros((max_degree, len(subset_lambdas)))
lasso_subset_lambda_boot_mse = np.zeros((max_degree, len(subset_lambdas)))
lasso_subset_lambda_boot_bias = np.zeros((max_degree, len(subset_lambdas)))
lasso_subset_lambda_boot_variance = np.zeros((max_degree, len(subset_lambdas)))

# Actual computations
for degree in range(max_degree):
    X = linear_regression.design_matrix_2D(x, y, degree)
    X_train = linear_regression.design_matrix_2D(x_train, y_train, degree)
    X_test = linear_regression.design_matrix_2D(x_test, y_test, degree)
    # Scaling and feeding to CV.
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    #    X_scaled[:,0] = 1 # Maybe not for ridge+lasso. Don't want to penalize constants...

    # Scaling and feeding to bootstrap and OLS
    scaler_boot = StandardScaler()
    scaler_boot.fit(X_train)
    X_train_scaled = scaler_boot.transform(X_train)
    X_test_scaled = scaler_boot.transform(X_test)
    #    X_train_scaled[:,0] = 1 #maybe not for ridge+lasso
    #    X_test_scaled[:,0] = 1 #maybe not for ridge+lasso

    # OLS, get MSE for test and train set.

    betas = linear_regression.OLS_SVD_2D(X_train_scaled, z_train)
    z_test_model = X_test_scaled @ betas
    z_train_model = X_train_scaled @ betas
    mse_ols_train[degree] = stat_tools.MSE(z_train, z_train_model)
    mse_ols_test[degree] = stat_tools.MSE(z_test, z_test_model)

    # CV, find best lambdas and get mse vs lambda for given degree. Also, gets
    # ols_CV_MSE

    lasso_cv_mse, ridge_cv_mse, ols_cv_mse_deg = crossvalidation.k_fold_cv_all(
        X_scaled, z, n_lambdas, lambdas, k_folds
    )
    best_lasso_lambda[degree] = lambdas[np.argmin(lasso_cv_mse)]
    best_ridge_lambda[degree] = lambdas[np.argmin(ridge_cv_mse)]
    best_lasso_mse[degree] = np.min(lasso_cv_mse)
    best_ridge_mse[degree] = np.min(ridge_cv_mse)
    lasso_lamb_deg_mse[degree] = lasso_cv_mse
    ridge_lamb_deg_mse[degree] = ridge_cv_mse
    ols_cv_mse[degree] = ols_cv_mse_deg

    # All regression bootstraps at once
    lamb_ridge = best_ridge_lambda[degree]
    lamb_lasso = best_lasso_lambda[degree]

    (
        ridge_mse,
        ridge_bias,
        ridge_variance,
        lasso_mse,
        lasso_bias,
        lasso_variance,
        ols_mse,
        ols_bias,
        ols_variance,
    ) = bootstrap.bootstrap_all(
        X_train_scaled, X_test_scaled, z_train, z_test, n_bootstraps, lamb_lasso, lamb_ridge
    )

    (
        ridge_best_lambda_boot_mse[degree],
        ridge_best_lambda_boot_bias[degree],
        ridge_best_lambda_boot_variance[degree],
    ) = (ridge_mse, ridge_bias, ridge_variance)

    (
        lasso_best_lambda_boot_mse[degree],
        lasso_best_lambda_boot_bias[degree],
        lasso_best_lambda_boot_variance[degree],
    ) = (lasso_mse, lasso_bias, lasso_variance)

    ols_boot_mse[degree], ols_boot_bias[degree], ols_boot_variance[degree] = (
        ols_mse,
        ols_bias,
        ols_variance,
    )

    # Bootstrapping for a selection of lambdas for ridge and lasso
    subset_lambda_index = 0
    for lamb in subset_lambdas:

        (
            ridge_mse,
            ridge_bias,
            ridge_variance,
            lasso_mse,
            lasso_bias,
            lasso_variance,
        ) = bootstrap.bootstrap_ridge_lasso(
            X_train_scaled, X_test_scaled, z_train, z_test, n_bootstraps, lamb_lasso, lamb_ridge
        )

        (
            ridge_subset_lambda_boot_mse[degree, subset_lambda_index],
            ridge_subset_lambda_boot_bias[degree, subset_lambda_index],
            ridge_subset_lambda_boot_variance[degree, subset_lambda_index],
        ) = (ridge_mse, ridge_bias, ridge_variance)

        (
            lasso_subset_lambda_boot_mse[degree, subset_lambda_index],
            lasso_subset_lambda_boot_bias[degree, subset_lambda_index],
            lasso_subset_lambda_boot_variance[degree, subset_lambda_index],
        ) = (lasso_mse, lasso_bias, lasso_variance)

        subset_lambda_index += 1

## See here, Lasso is best when you actually CAN discard predictors. Note the issues
## in OLS if more predictors than responses.

plt.subplot(131)
plt.semilogy(lasso_best_lambda_boot_mse, label="MSE")
plt.semilogy(lasso_best_lambda_boot_bias, label="Bias$^2$")
plt.semilogy(lasso_best_lambda_boot_variance, label="Var")
plt.xlabel("Complexity")
plt.legend()
plt.title("LASSO")


plt.subplot(132)
plt.semilogy(ridge_best_lambda_boot_mse, label="MSE")
plt.semilogy(ridge_best_lambda_boot_bias, label="Bias$^2$")
plt.semilogy(ridge_best_lambda_boot_variance, label="Var")
plt.xlabel("Complexity")
plt.legend()
plt.title("Ridge")

plt.subplot(133)
plt.semilogy(ols_boot_mse, label="MSE")
plt.semilogy(ols_boot_bias, label="Bias$^2$")
plt.semilogy(ols_boot_variance, label="Var")
plt.xlabel("Complexity")
plt.legend()
plt.title("bias-variance ols")
plt.show()

plt.subplot(121)
plt.plot(np.log10(lambdas), lasso_lamb_deg_mse[1], label="Deg=1")
plt.plot(np.log10(lambdas), lasso_lamb_deg_mse[3], label="Deg=3")
plt.plot(np.log10(lambdas), lasso_lamb_deg_mse[6], label="Deg=6")
plt.plot(np.log10(lambdas), lasso_lamb_deg_mse[9], label="Deg=9")
plt.plot(np.log10(lambdas), lasso_lamb_deg_mse[12], label="Deg=12")
plt.title("selected degrees lasso lambda vs mse")

plt.subplot(122)
plt.plot(np.log10(lambdas), ridge_lamb_deg_mse[1], label="Deg=1")
plt.plot(np.log10(lambdas), ridge_lamb_deg_mse[3], label="Deg=3")
plt.plot(np.log10(lambdas), ridge_lamb_deg_mse[6], label="Deg=6")
plt.plot(np.log10(lambdas), ridge_lamb_deg_mse[9], label="Deg=9")
plt.plot(np.log10(lambdas), ridge_lamb_deg_mse[12], label="Deg=12")
plt.title("selected degrees ridge lambda vs mse")
plt.show()

print("best lasso lambdas:")
print(best_lasso_lambda)
print("best ridge lambdas:")
print(best_ridge_lambda)


# Note how the lasso coeff include higher order x's, while managing to kill the y's.
# Also note, that a column of zeros is included leftmost in the design matrix due to
# centering of the responses. This should always give beta[0] = 0.
clf_best_lasso = skl.Lasso(alpha=best_lasso_lambda[max_degree - 1], fit_intercept=False).fit(
    X_train_scaled, z_train
)
print("betas from lasso for highest degree in test, and best lambda for that degree")
print(clf_best_lasso.coef_)

clf_best_ridge = skl.Ridge(alpha=best_ridge_lambda[max_degree - 1], fit_intercept=False).fit(
    X_train_scaled, z_train
)
print("betas from ridge for highest degree in test, and best lambda for that degree")
print(clf_best_ridge.coef_)
