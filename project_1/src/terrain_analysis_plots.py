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


def terrain_analysis_plots(spacing=40, max_degree = 25, n_lambdas=30, k_folds = 5, n_bootstraps=50, do_boot=False, do_subset=False):

    # Setting up the terrain data:
    # Note structure! X-coordinates are on the rows of terrain_data
    # Point_selection.flatten() moves most rapidly over the x-coordinates
    # Meshgrids flattened also move most rapidly over the x-coordinates. Thus
    # this should make z(x,y).reshape(length_y,length_x) be consistent with terrain_data

    terrain_data = imread('../datafiles/SRTM_data_Norway_1.tif')
    point_selection = terrain_data[:1801:spacing,:1801:spacing] # Make square and downsample
    x_terrain_selection = np.linspace(0,1,point_selection.shape[1])
    y_terrain_selection = np.linspace(0,1,point_selection.shape[0])
    X_coord_selection, Y_coord_selection = np.meshgrid(x_terrain_selection, y_terrain_selection)
    z_terrain_selection = point_selection.flatten() # the response values
    x_terrain_selection_flat = X_coord_selection.flatten() # the first degree feature variables
    y_terrain_selection_flat = Y_coord_selection.flatten() # the first degree feature variables


    lambdas = np.logspace(-5,0,n_lambdas)
    subset_lambdas = lambdas[::15]


    x = x_terrain_selection_flat
    y = y_terrain_selection_flat
    z = z_terrain_selection


    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size = 0.2)

    # Centering
    z_intercept = np.mean(z)
    z = z - z_intercept

    z_train_intercept = np.mean(z_train)
    z_train = z_train - z_train_intercept
    z_test = z_test - z_train_intercept


    ##### Setup of problem is completede above.


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
    #    X_scaled[:,0] = 1 # Probably should not have this.


        # Scaling and feeding to bootstrap and OLS
        scaler_boot = StandardScaler()
        scaler_boot.fit(X_train)
        X_train_scaled = scaler_boot.transform(X_train)
        X_test_scaled = scaler_boot.transform(X_test)
    #    X_train_scaled[:,0] = 1 # Probably actually not
    #    X_test_scaled[:,0] = 1 # Have a bad feeling about how this might affect ridge/lasso.



        # OLS, get MSE for test and train set.

        betas = linear_regression.OLS_SVD_2D(X_train_scaled, z_train)
        z_test_model = X_test_scaled @ betas
        z_train_model = X_train_scaled @ betas
        mse_ols_train[degree] = stat_tools.MSE(z_train, z_train_model)
        mse_ols_test[degree] = stat_tools.MSE(z_test, z_test_model)


        # CV, find best lambdas and get mse vs lambda for given degree.

        lasso_cv_mse, ridge_cv_mse, ols_cv_mse_deg = stat_tools.k_fold_cv_all(X_scaled,z,n_lambdas,lambdas,k_folds)
        best_lasso_lambda[degree] = lambdas[np.argmin(lasso_cv_mse)]
        best_ridge_lambda[degree] = lambdas[np.argmin(ridge_cv_mse)]
        best_lasso_mse[degree] = np.min(lasso_cv_mse)
        best_ridge_mse[degree] = np.min(ridge_cv_mse)
        lasso_lamb_deg_mse[degree] = lasso_cv_mse
        ridge_lamb_deg_mse[degree] = ridge_cv_mse
        ols_cv_mse[degree] = ols_cv_mse_deg


        if do_boot:
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

        if do_subset:
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



    # Plots go here.


    plt.figure()
    plt.semilogy(ols_cv_mse)
    plt.semilogy(best_ridge_mse)
    plt.semilogy(best_lasso_mse)
    plt.title('CV MSE for OLS, Ridge and Lasso, with the best lambdas for each degree')

    return
if __name__ == '__main__':
    terrain_analysis_plots()
