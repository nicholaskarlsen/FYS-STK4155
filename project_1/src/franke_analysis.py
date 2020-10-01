import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import linear_regression
import utils
import stat_tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as skl
from imageio import imread

import stat_tools

utils.plot_settings() # LaTeX fonts in Plots!

def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


n = 500
noise_scale = 0.2
x = np.random.uniform(0, 1, n)
y = np.random.uniform(0, 1, n)
z = FrankeFunction(x, y)
# Adding standard normal noise:
z = z + noise_scale*np.random.normal(0,1,len(z))
max_degree = 15
n_lambdas = 30
n_bootstraps = 100
k_folds = 5
lambdas = np.logspace(-5,0,n_lambdas)
subset_lambdas = lambdas[::60]

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size = 0.2)

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
    X = linear_regression.design_matrix_2D(x,y,degree)
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

    lasso_cv_mse, ridge_cv_mse, ols_cv_mse_deg = stat_tools.k_fold_cv_all(X_scaled,z,n_lambdas,lambdas,k_folds)
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

    ridge_mse, ridge_bias, ridge_variance, lasso_mse, lasso_bias, lasso_variance, ols_mse, ols_bias, ols_variance = \
    stat_tools.bootstrap_all(X_train_scaled, X_test_scaled, z_train, z_test, n_bootstraps, lamb_lasso, lamb_ridge)

    ridge_best_lambda_boot_mse[degree], ridge_best_lambda_boot_bias[degree], \
    ridge_best_lambda_boot_variance[degree] = ridge_mse, ridge_bias, ridge_variance

    lasso_best_lambda_boot_mse[degree], lasso_best_lambda_boot_bias[degree], \
    lasso_best_lambda_boot_variance[degree] = lasso_mse, lasso_bias, lasso_variance

    ols_boot_mse[degree], ols_boot_bias[degree], \
    ols_boot_variance[degree] = ols_mse, ols_bias, ols_variance

    # Bootstrapping for a selection of lambdas for ridge and lasso
    # subset_lambda_index = 0
    # for lamb in subset_lambdas:
    #
    #     ridge_mse, ridge_bias, ridge_variance, lasso_mse, lasso_bias, lasso_variance = \
    #     stat_tools.bootstrap_ridge_lasso(X_train_scaled, X_test_scaled, z_train, z_test, n_bootstraps, lamb_lasso, lamb_ridge)
    #
    #     ridge_subset_lambda_boot_mse[degree, subset_lambda_index ], ridge_subset_lambda_boot_bias[degree, subset_lambda_index ], \
    #     ridge_subset_lambda_boot_variance[degree, subset_lambda_index ] = ridge_mse, ridge_bias, ridge_variance
    #
    #     lasso_subset_lambda_boot_mse[degree, subset_lambda_index ], lasso_subset_lambda_boot_bias[degree, subset_lambda_index ], \
    #     lasso_subset_lambda_boot_variance[degree, subset_lambda_index ] = lasso_mse, lasso_bias, lasso_variance
    #
    #     subset_lambda_index  += 1

# Plots go here. Point is to use the previous computations to obtain the best hyper-parameters (lambda, degree)
# for the different regression methods.




########### Final comparison with ground truth:
# Do best parameters (lambda, degree) plots of uniformly sampled x,y grid here.
# Aim is to compare the Franke function evaluated at that grid, with the best of the
# 3 regression methods (OLS, Ridge, Lasso). The methods will be trained on the training set X_train.
# These trainings will produce betas. These betas will be applied to a (scaled) x,y-grid design matrix
# and the z_train_intercept will be added to the result.

# OLS

degree = 15

X = linear_regression.design_matrix_2D(x,y,degree)
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
betas = linear_regression.OLS_SVD_2D(X_scaled, z)


x_plot = np.linspace(0,1,2000)
y_plot = np.linspace(0,1,2000)
x_plot_mesh, y_plot_mesh = np.meshgrid(x_plot,y_plot)
x_plot_mesh_flat, y_plot_mesh_flat = x_plot_mesh.flatten(), y_plot_mesh.flatten()
z_plot_franke = FrankeFunction(x_plot_mesh, y_plot_mesh)
X_plot_design = linear_regression.design_matrix_2D(x_plot_mesh_flat,y_plot_mesh_flat,degree)
X_plot_design_scaled = scaler.transform(X_plot_design)
z_predict_flat = (X_plot_design_scaled @ betas) + z_intercept



fig = plt.figure()

# Plot the analytic curve
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.set_title("Franke Function")
ax.view_init(azim=45)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
surf = ax.plot_surface(x_plot_mesh, y_plot_mesh, z_plot_franke, cmap=cm.coolwarm)

# Plot the fitted curve
ax = fig.add_subplot(1, 2, 2, projection="3d")
ax.set_title("OLS")
ax.view_init(azim=45)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
surf = ax.plot_surface(x_plot_mesh, y_plot_mesh, z_predict_flat.reshape(-1,2000), cmap=cm.coolwarm)



# Ridge

degree = 15
ridge_lambda = 1e-2

X = linear_regression.design_matrix_2D(x,y,degree)
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
betas_ridge = linear_regression.Ridge_2D(X_scaled, z, ridge_lambda)


x_plot = np.linspace(0,1,2000)
y_plot = np.linspace(0,1,2000)
x_plot_mesh, y_plot_mesh = np.meshgrid(x_plot,y_plot)
x_plot_mesh_flat, y_plot_mesh_flat = x_plot_mesh.flatten(), y_plot_mesh.flatten()
z_plot_franke = FrankeFunction(x_plot_mesh, y_plot_mesh)
X_plot_design = linear_regression.design_matrix_2D(x_plot_mesh_flat,y_plot_mesh_flat,degree)
X_plot_design_scaled = scaler.transform(X_plot_design)
z_predict_flat = (X_plot_design_scaled @ betas_ridge) + z_intercept


fig = plt.figure()

# Plot the analytic curve
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.set_title("Franke Function")
ax.view_init(azim=45)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
surf = ax.plot_surface(x_plot_mesh, y_plot_mesh, z_plot_franke, cmap=cm.coolwarm)

# Plot the fitted curve
ax = fig.add_subplot(1, 2, 2, projection="3d")
ax.set_title("Ridge")
ax.view_init(azim=45)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
surf = ax.plot_surface(x_plot_mesh, y_plot_mesh, z_predict_flat.reshape(2000,-1), cmap=cm.coolwarm)


# Lasso

degree = 15
lasso_lambda = 1e-5

X = linear_regression.design_matrix_2D(x,y,degree)
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
clf_Lasso = skl.Lasso(alpha=lasso_lambda,fit_intercept=False).fit(X_scaled,z)


x_plot = np.linspace(0,1,2000)
y_plot = np.linspace(0,1,2000)
x_plot_mesh, y_plot_mesh = np.meshgrid(x_plot,y_plot)
x_plot_mesh_flat, y_plot_mesh_flat = x_plot_mesh.flatten(), y_plot_mesh.flatten()
z_plot_franke = FrankeFunction(x_plot_mesh, y_plot_mesh)
X_plot_design = linear_regression.design_matrix_2D(x_plot_mesh_flat,y_plot_mesh_flat,degree)
X_plot_design_scaled = scaler.transform(X_plot_design)
z_predict_flat = clf_Lasso.predict(X_plot_design_scaled) + z_intercept

fig = plt.figure()

# Plot the analytic curve
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.set_title("Franke Function")
ax.view_init(azim=45)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
surf = ax.plot_surface(x_plot_mesh, y_plot_mesh, z_plot_franke, cmap=cm.coolwarm)

# Plot the fitted curve
ax = fig.add_subplot(1, 2, 2, projection="3d")
ax.set_title("Lasso")
ax.view_init(azim=45)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
surf = ax.plot_surface(x_plot_mesh, y_plot_mesh, z_predict_flat.reshape(2000,-1), cmap=cm.coolwarm)
