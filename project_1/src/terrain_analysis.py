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

utils.plot_settings() # LaTeX fonts in Plots!


def terrain_analysis():
    # Setting up the terrain data:
    terrain_data = imread('../datafiles/SRTM_data_Norway_1.tif')
    x_terrain = np.arange(terrain_data.shape[1])
    y_terrain = np.arange(terrain_data.shape[0])
    X_coord, Y_coord = np.meshgrid(x_terrain,y_terrain)
    z_terrain = terrain_data.flatten() # the response values
    x_terrain_flat = X_coord.flatten() # the first degree feature variables
    y_terrain_flat = Y_coord.flatten() # the first degree feature variables

    max_degree = 20
    n_lambdas = 30
    n_bootstraps = 50
    k_folds = 5
    lambdas = np.logspace(-3,0,n_lambdas)
    subset_lambdas = lambdas[::5]

    #### Should select a subset in some manner of the terrain points
    #### Should probably also make the feature variables be float that range from [0,1]

    x = x_terrain_flat[::20]
    y = y_terrain_flat[::20]
    z = z_terrain[::20]


    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size = 0.2)

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
        X_scaled[:,0] = 1 # Probably should not have this.

        # Scaling and feeding to bootstrap and OLS
        scaler_boot = StandardScaler()
        scaler_boot.fit(X_train)
        X_train_scaled = scaler_boot.transform(X_train)
        X_test_scaled = scaler_boot.transform(X_test)
        X_train_scaled[:,0] = 1 # Probably actually not
        X_test_scaled[:,0] = 1 # Have a bad feeling about how this might affect ridge/lasso.

        # OLS, get MSE for test and train set.

        betas = linear_regression.OLS_SVD_2D(X_train_scaled, z_train)
        z_test_model = X_test_scaled @ betas
        z_train_model = X_train_scaled @ betas
        mse_ols_train[degree] = stat_tools.MSE(z_train, z_train_model)
        mse_ols_test[degree] = stat_tools.MSE(z_test, z_test_model)


        # CV, find best lambdas and get mse vs lambda for given degree.

        lasso_cv_mse, ridge_cv_mse, ols_cv_mse = stat_tools.k_fold_cv_all(X_scaled,z,n_lambdas,lambdas,k_folds)
        best_lasso_lambda[degree] = lambdas[np.argmin(lasso_cv_mse)]
        best_ridge_lambda[degree] = lambdas[np.argmin(ridge_cv_mse)]
        best_lasso_mse[degree] = np.min(lasso_cv_mse)
        best_ridge_mse[degree] = np.min(ridge_cv_mse)
        lasso_lamb_deg_mse[degree] = lasso_cv_mse
        ridge_lamb_deg_mse[degree] = ridge_cv_mse

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
        subset_lambda_index = 0
        for lamb in subset_lambdas:

            ridge_mse, ridge_bias, ridge_variance, lasso_mse, lasso_bias, lasso_variance = \
            stat_tools.bootstrap_ridge_lasso(X_train_scaled, X_test_scaled, z_train, z_test, n_bootstraps, lamb_lasso, lamb_ridge)

            ridge_subset_lambda_boot_mse[degree, subset_lambda_index ], ridge_subset_lambda_boot_bias[degree, subset_lambda_index ], \
            ridge_subset_lambda_boot_variance[degree, subset_lambda_index ] = ridge_mse, ridge_bias, ridge_variance

            lasso_subset_lambda_boot_mse[degree, subset_lambda_index ], lasso_subset_lambda_boot_bias[degree, subset_lambda_index ], \
            lasso_subset_lambda_boot_variance[degree, subset_lambda_index ] = lasso_mse, lasso_bias, lasso_variance

            subset_lambda_index  += 1



################ All necessary computations should have been done above. Below follows
################ the plotting part.

    return
