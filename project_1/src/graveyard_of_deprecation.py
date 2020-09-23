######### DEPRECATED ############



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

def bootstrap_all(X_train, X_test, z_train, z_test, n_bootstraps, lamb_lasso, lamb_ridge):
    """ Performs the bootstrapped bias variance analysis for OLS, Ridge and Lasso, given input
        training and test data, the number of bootstrap iterations and the lambda values for
        Ridge and Lasso.

        Returns MSE, mean squared bias and mean variance for Ridge, Lasso and OLS in that order.
    """

    z_boot_ols = np.zeros((len(z_test),n_bootstraps))
    z_boot_ridge = np.zeros((len(z_test),n_bootstraps))
    z_boot_lasso= np.zeros((len(z_test),n_bootstraps))
    for i in range(n_bootstraps):
        shuffle = np.random.randint(0,len(z_train),len(z_train))
        X_boot, z_boot = X_train[shuffle] , z_train[shuffle]
        betas_boot_ols = linear_regression.OLS_SVD_2D(X_boot, z_boot)
        betas_boot_ridge = linear_regression.Ridge_2D(X_boot, z_boot, lamb_ridge) #Ridge, given lambda
        clf_Lasso = skl.Lasso(alpha=lamb_lasso,fit_intercept=False).fit(X_boot,z_boot)
        z_boot_lasso[:,i] = clf_Lasso.predict(X_test) #Lasso, given lambda
        z_boot_ridge[:,i] = X_test @ betas_boot_ridge
        z_boot_ols[:,i] = X_test @ betas_boot_ols

    ridge_mse, ridge_bias, ridge_variance = stat_tools.compute_mse_bias_variance(z_test, z_boot_ridge)

    lasso_mse, lasso_bias, lasso_variance = stat_tools.compute_mse_bias_variance(z_test, z_boot_lasso)

    ols_mse, ols_bias, ols_variance = stat_tools.compute_mse_bias_variance(z_test, z_boot_ols)

    return ridge_mse, ridge_bias, ridge_variance, lasso_mse, lasso_bias, lasso_variance, ols_mse, ols_bias, ols_variance

def bootstrap_ridge_lasso(X_train, X_test, z_train, z_test, n_bootstraps, lamb_lasso, lamb_ridge):
    """ Performs the bootstrapped bias variance analysis for only Ridge and Lasso, given input
        training and test data, the number of bootstrap iterations and the lambda values for
        Ridge and Lasso. Intended for studying bias/variance dependency as a function of lambda-values.

        Returns MSE, mean squared bias and mean variance for Ridge and Lasso, in that order
    """

    z_boot_ridge = np.zeros((len(z_test),n_bootstraps))
    z_boot_lasso= np.zeros((len(z_test),n_bootstraps))
    for i in range(n_bootstraps):
        shuffle = np.random.randint(0,len(z_train),len(z_train))
        X_boot, z_boot = X_train[shuffle] , z_train[shuffle]
        betas_boot_ridge = linear_regression.Ridge_2D(X_boot, z_boot, lamb_ridge) #Ridge, given lambda
        clf_Lasso = skl.Lasso(alpha=lamb_lasso,fit_intercept=False).fit(X_boot,z_boot)
        z_boot_lasso[:,i] = clf_Lasso.predict(X_test) #Lasso, given lambda
        z_boot_ridge[:,i] = X_test @ betas_boot_ridge

    ridge_mse, ridge_bias, ridge_variance = stat_tools.compute_mse_bias_variance(z_test, z_boot_ridge)

    lasso_mse, lasso_bias, lasso_variance = stat_tools.compute_mse_bias_variance(z_test, z_boot_lasso)

    return ridge_mse, ridge_bias, ridge_variance, lasso_mse, lasso_bias, lasso_variance



def k_fold_cv_all(X,z,n_lambdas,lambdas,k_folds):
    """ Performs k-fold cross validation for Ridge, Lasso and OLS. The Lasso and Ridge
        MSE-values are computed for a number of n_lambdas, with the lambda values given
        by the lambdas array. OLS is done only once for each of the k_folds folds.

        Args:
            X (array): Design matrix
            z (array): Data-values/response-values/whatever-they-are-called-in-your-field-values
            n_lambdas (int): number of lambda values to use for Lasso and Ridge.
            lambdas (array): The actual lambda-values to try.
            k_folds (int): The number of folds.

        Return:
            lasso_cv_mse (array): array containing the computed MSE for each lambda in Lasso
            ridge_cv_mse (array): array containing the computed MSE for each lambda in Ridge
            ols_cv_mse (float): computed MSE for OLS.
    """



    ridge_fold_score = np.zeros((n_lambdas, k_folds))
    lasso_fold_score = np.zeros((n_lambdas, k_folds))
    test_list, train_list = stat_tools.k_fold_selection(z, k_folds)
    for i in range(n_lambdas):
        lamb = lambdas[i]
        for j in range(k_folds):
            test_ind_cv = test_list[j]
            train_ind_cv = train_list[j]
            X_train_cv = X[train_ind_cv]
            z_train_cv = z[train_ind_cv]
            X_test_cv = X[test_ind_cv]
            z_test_cv = z[test_ind_cv]
            clf_Lasso = skl.Lasso(alpha=lamb,fit_intercept=False).fit(X_train_cv,z_train_cv)
            z_lasso_test = clf_Lasso.predict(X_test_cv)
            ridge_betas = linear_regression.Ridge_2D(X_train_cv, z_train_cv, lamb)
            z_ridge_test = X_test_cv @ ridge_betas
            ridge_fold_score[i,j] = stat_tools.MSE(z_test_cv, z_ridge_test)
            lasso_fold_score[i,j] = stat_tools.MSE(z_test_cv, z_lasso_test)

    lasso_cv_mse = np.mean(lasso_fold_score, axis=1)
    ridge_cv_mse = np.mean(ridge_fold_score, axis=1)

    # Get ols_mse for cv.
    ols_fold_score = np.zeros(k_folds)
    for i in range(k_folds):
        test_ind_cv = test_list[j]
        train_ind_cv = train_list[j]
        X_train_cv = X[train_ind_cv]
        z_train_cv = z[train_ind_cv]
        X_test_cv = X[test_ind_cv]
        z_test_cv = z[test_ind_cv]
        ols_cv_betas = linear_regression.OLS_SVD_2D(X_train_cv, z_train_cv)
        z_ols_test = X_test_cv @ ols_cv_betas
        ols_fold_score[i] = stat_tools.MSE(z_test_cv, z_ols_test)

    ols_cv_mse = np.mean(ols_fold_score)


    return lasso_cv_mse, ridge_cv_mse, ols_cv_mse


def k_folds_cv_OLS_only(X,z,k_folds):
    """ As could be guessed, computes the k-fold cross-validation MSE for OLS, given
        input X, y as data; k_folds as number of folds. Returns the computed MSE.

    """


    ols_fold_score = np.zeros(k_folds)
    test_list, train_list = stat_tools.k_fold_selection(z, k_folds)
    for i in range(k_folds):
        test_ind_cv = test_list[j]
        train_ind_cv = train_list[j]
        X_train_cv = X[train_ind_cv]
        z_train_cv = z[train_ind_cv]
        X_test_cv = X[test_ind_cv]
        z_test_cv = z[test_ind_cv]
        ols_cv_betas = linear_regression.OLS_SVD_2D(X_train_cv, z_train_cv)
        z_ols_test = X_test_cv @ ols_cv_betas
        ols_fold_score[i] = stat_tools.MSE(z_test_cv, z_ols_test)

    ols_cv_mse = np.mean(ols_fold_score)

    return ols_cv_mse


def deprecated_franke_analysis():

    n = 100
    noise_scale = 0.2
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    z = FrankeFunction(x, y)
    # Adding standard normal noise:
    z = z + noise_scale*np.random.normal(0,1,len(z))
    max_degree = 10
    n_lambdas = 30
    n_bootstraps = 50
    k_folds = 5
    lambdas = np.logspace(-3,0,n_lambdas)
    subset_lambdas = lambdas[::5]

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

    ridge_subset_lambda_boot_mse = np.zeros((max_degree, subset_lambdas))
    ridge_subset_lambda_boot_bias = np.zeros((max_degree, subset_lambdas))
    ridge_subset_lambda_boot_variance = np.zeros()(max_degree, subset_lambdas))
    lasso_subset_lambda_boot_mse = np.zeros((max_degree, subset_lambdas))
    lasso_subset_lambda_boot_bias = np.zeros((max_degree, subset_lambdas))
    lasso_subset_lambda_boot_variance = np.zeros((max_degree, subset_lambdas))

    # Actual computations
    for degree in range(max_degree):
        X = linear_regression.design_matrix_2D(x,y,degree)
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2)
        # Scaling and feeding to CV.
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X[:,0] = 1

        # Scaling and feeding to bootstrap and OLS
        scaler_boot = StandardScaler()
        scaler_boot.fit(X_train)
        X_train_scaled = scaler_boot.transform(X_train)
        X_test_scaled = scaler_boot.transform(X_test)
        X_train_scaled[:,0] = 1
        X_test_scaled[:,0] = 1

        # OLS, get MSE for test and train set.

        betas = linear_regression.OLS_SVD_2D(X_train_scaled, z_train)
        z_test_model = X_test_scaled @ betas
        z_train_model = X_train_scaled @ betas
        mse_ols_train[degree] = stat_tools.MSE(z_train, z_train_model)
        mse_ols_test[degree] = stat_tools.MSE(z_test, z_test_model)


        # CV, find best lambdas and get mse vs lambda for given degree.

        ridge_fold_score = np.zeros((n_lambdas, k_folds))
        lasso_fold_score = np.zeros((n_lambdas, k_folds))
        test_list, train_list = stat_tools.k_fold_selection(z, k_folds)
        for i in range(n_lambdas):
            lamb = lambdas[i]
            for j in range(k_folds):
                test_ind_cv = test_list[j]
                train_ind_cv = train_list[j]
                X_train_cv = X[train_ind_cv]
                z_train_cv = z[train_ind_cv]
                X_test_cv = X[test_ind_cv]
                z_test_cv = z[test_ind_cv]
                clf_Lasso = skl.Lasso(alpha=lamb,fit_intercept=False).fit(X_train_cv,z_train_cv)
                z_lasso_test = clf_Lasso.predict(X_test_cv)
                ridge_betas = linear_regression.Ridge_2D(X_train_cv, z_train_cv, lamb)
                z_ridge_test = X_test_cv @ ridge_betas
                ridge_fold_score[i,j] = stat_tools.MSE(z_test_cv, z_ridge_test)
                lasso_fold_score[i,j] = stat_tools.MSE(z_test_cv, z_lasso_test)

        lasso_cv_mse = np.mean(lasso_fold_score, axis=1)
        ridge_cv_mse = np.mean(ridge_fold_score, axis=1)
        best_lasso_lambda[degree] = lambdas[np.argmin(lasso_cv_mse)]
        best_ridge_lambda[degree] = lambdas[np.argmin(ridge_cv_mse)]
        best_lasso_mse[degree] = np.min(lasso_cv_mse)
        best_ridge_mse[degree] = np.min(ridge_cv_mse)
        lasso_lamb_deg_mse[degree] = lasso_cv_mse
        ridge_lamb_deg_mse[degree] = ridge_cv_mse

        # # Get ols_mse for cv.
        # ols_fold_score = np.zeros(k_folds)
        # for i in range(k_folds):
        #     test_ind_cv = test_list[j]
        #     train_ind_cv = train_list[j]
        #     X_train_cv = X[train_ind_cv]
        #     z_train_cv = z[train_ind_cv]
        #     X_test_cv = X[test_ind_cv]
        #     z_test_cv = z[test_ind_cv]
        #     ols_cv_betas = linear_regression.OLS_SVD_2D(X_train_cv, z_train_cv)
        #     z_ols_test = X_test_cv @ ols_cv_betas
        #     ols_fold_score[i] = stat_tools.MSE(z_test_cv, z_ols_test)
        #
        # ols_cv_mse = np.mean(ols_fold_score)
        #
        # # OLS bootstap, get bootstrapped mse, bias and variance for given degree.
        # z_boot_model = np.zeros((len(z_test),n_bootstraps))
        # for i in range(n_bootstraps):
        #     shuffle = np.random.randint(0,len(z_train),len(z_train))
        #     X_boot, z_boot = X_train_scaled[shuffle] , z_train[shuffle]
        #     betas_boot = linear_regression.OLS_SVD_2D(X_boot, z_boot)
        #     #betas_boot = linear_regression.Ridge_2D(X_boot, z_boot, lamb) #Ridge, given lambda
        #     #clf_Lasso = skl.Lasso(alpha=lamb).fit(X_boot,z_boot)
        #     #z_boot_model[:,i] = clf_Lasso_predict(X_test) #Lasso, given lambda
        #     z_boot_model[:,i] = X_test_scaled @ betas_boot
        # mse, bias, variance = stat_tools.compute_mse_bias_variance(z_test, z_boot_model)
        # ols_boot_mse[degree] = mse
        # ols_boot_bias[degree] = bias
        # ols_boot_variance[degree] = variance
        #
        # # Ridge bootstrap, get bootstrapped mse, bias and variance for given degree and lambda
        # lamb = best_ridge_lambda[degree]
        # z_boot_model = np.zeros((len(z_test),n_bootstraps))
        # for i in range(n_bootstraps):
        #     shuffle = np.random.randint(0,len(z_train),len(z_train))
        #     X_boot, z_boot = X_train_scaled[shuffle] , z_train[shuffle]
        #     #betas_boot = linear_regression.OLS_SVD_2D(X_boot, z_boot)
        #     betas_boot = linear_regression.Ridge_2D(X_boot, z_boot, lamb) #Ridge, given lambda
        #     #clf_Lasso = skl.Lasso(alpha=lamb).fit(X_boot,z_boot)
        #     #z_boot_model[:,i] = clf_Lasso_predict(X_test) #Lasso, given lambda
        #     z_boot_model[:,i] = X_test_scaled @ betas_boot
        # mse, bias, variance = stat_tools.compute_mse_bias_variance(z_test, z_boot_model)
        # ridge_best_lambda_boot_mse[degree] = mse
        # ridge_best_lambda_boot_bias[degree] = bias
        # ridge_best_lambda_boot_variance[degree] = variance
        #
        # # Lasso bootstrap, get bootstrapped mse, bias and variance for given degree and lambda.
        # lamb = best_lasso_lambda[degree]
        # z_boot_model = np.zeros((len(z_test),n_bootstraps))
        # for i in range(n_bootstraps):
        #     shuffle = np.random.randint(0,len(z_train),len(z_train))
        #     X_boot, z_boot = X_train_scaled[shuffle] , z_train[shuffle]
        #     #betas_boot = linear_regression.OLS_SVD_2D(X_boot, z_boot)
        #     #betas_boot = linear_regression.Ridge_2D(X_boot, z_boot, lamb) #Ridge, given lambda
        #     clf_Lasso = skl.Lasso(alpha=lamb,fit_intercept=False).fit(X_boot,z_boot)
        #     z_boot_model[:,i] = clf_Lasso.predict(X_test_scaled) #Lasso, given lambda
        #     #z_boot_model[:,i] = X_test_scaled @ betas_boot
        # mse, bias, variance = stat_tools.compute_mse_bias_variance(z_test, z_boot_model)
        # lasso_best_lambda_boot_mse[degree] = mse
        # lasso_best_lambda_boot_bias[degree] = bias
        # lasso_best_lambda_boot_variance[degree] = variance

        # All regressions bootstraps at once
        lamb_ridge = best_ridge_lambda[degree]
        lamb_lasso = best_lasso_lambda[degree]
        z_boot_ols = np.zeros((len(z_test),n_bootstraps))
        z_boot_ridge = np.zeros((len(z_test),n_bootstraps))
        z_boot_lasso= np.zeros((len(z_test),n_bootstraps))
        for i in range(n_bootstraps):
            shuffle = np.random.randint(0,len(z_train),len(z_train))
            X_boot, z_boot = X_train_scaled[shuffle] , z_train[shuffle]
            betas_boot_ols = linear_regression.OLS_SVD_2D(X_boot, z_boot)
            betas_boot_ridge = linear_regression.Ridge_2D(X_boot, z_boot, lamb_ridge) #Ridge, given lambda
            clf_Lasso = skl.Lasso(alpha=lasso_lamb,fit_intercept=False).fit(X_boot,z_boot)
            z_boot_lasso[:,i] = clf_Lasso.predict(X_test_scaled) #Lasso, given lambda
            z_boot_ridge[:,i] = X_test_scaled @ betas_boot_ridge
            z_boot_ols[:,i] = X_test_scaled @ betas_boot_ols

        ridge_best_lambda_boot_mse[degree], ridge_best_lambda_boot_bias[degree], \
        ridge_best_lambda_boot_variance[degree] = stat_tools.compute_mse_bias_variance(z_test, z_boot_ridge)

        lasso_best_lambda_boot_mse[degree], lasso_best_lambda_boot_bias[degree], \
        lasso_best_lambda_boot_variance[degree] = stat_tools.compute_mse_bias_variance(z_test, z_boot_lasso)

        ols_boot_mse[degree], ols_boot_bias[degree], \
        ols_boot_variance[degree] = stat_tools.compute_mse_bias_variance(z_test, z_boot_ols)

        # Bootstrapping for a selection of lambdas for ridge and lasso
        i = 0
        for lamb in subset_lambdas:
            z_boot_ridge = np.zeros((len(z_test),n_bootstraps))
            z_boot_lasso= np.zeros((len(z_test),n_bootstraps))
            for i in range(n_bootstraps):
                shuffle = np.random.randint(0,len(z_train),len(z_train))
                X_boot, z_boot = X_train_scaled[shuffle] , z_train[shuffle]
                betas_boot_ridge = linear_regression.Ridge_2D(X_boot, z_boot, lamb) #Ridge, given lambda
                clf_Lasso = skl.Lasso(alpha=lamb,fit_intercept=False).fit(X_boot,z_boot)
                z_boot_lasso[:,i] = clf_Lasso.predict(X_test_scaled) #Lasso, given lambda
                z_boot_ridge[:,i] = X_test_scaled @ betas_boot_ridge

            ridge_subset_lambda_boot_mse[degree, i], ridge_subset_lambda_boot_bias[degree, i], \
            ridge_subset_lambda_boot_variance[degree, i] = stat_tools.compute_mse_bias_variance(z_test, z_boot_ridge)

            lasso_subset_lambda_boot_mse[degree, i], lasso_subset_lambda_boot_bias[degree, i], \
            lasso_subset_lambda_boot_variance[degree, i] = stat_tools.compute_mse_bias_variance(z_test, z_boot_lasso)

            i = i+1

    return

def terrain_analysis():
    # Setting up the terrain data:
    terrain_data = imread('../datafiles/SRTM_data_Norway_1.tif')
    x_terrain = np.arange(terrain_data.shape[1]) #apparently, from the problem description.
    y_terrain = np.arange(terrain_data.shape[0])
    X_coord, Y_coord = np.meshgrid(x_terrain,y_terrain)
    z_terrain = terrain_data.flatten() # the response values
    x_terrain_flat = X_coord.flatten() # the first degree feature variables
    y_terrain_flat = Y_coord.flatten() # the first degree feature variables
    max_degree = 10
    n_lambdas = 15
    n_bootstraps = 20
    k_folds = 5
    lambdas = np.logspace(-3,0,n_lambdas)

    # Quantities of interest:
    mse_ols_test = np.zeros(max_degree)
    mse_ols_train = np.zeros(max_degree)

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

    # Actual computations
    for degree in range(max_degree):
        X_terrain_design = linear_regression.design_matrix_2D(x_terrain_flat,y_terrain_flat,degree)
        X_train, X_test, z_train, z_test = train_test_split(X_terrain_design, z_terrain, test_size = 0.2)
        # Scaling and feeding to CV.
        z = z_terrain
        X = X_terrain_design
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X[:,0] = 1

        # Scaling and feeding to bootstrap and OLS
        scaler_boot = StandardScaler()
        scaler_boot.fit(X_train)
        X_train_scaled = scaler_boot.transform(X_train)
        X_test_scaled = scaler_boot.transform(X_test)
        X_train_scaled[:,0] = 1
        X_test_scaled[:,0] = 1

        # OLS, get MSE for test and train set.

        betas = linear_regression.OLS_SVD_2D(X_train_scaled, z_train)
        z_test_model = X_test_scaled @ betas
        z_train_model = X_train_scaled @ betas
        mse_ols_train[degree] = stat_tools.MSE(z_train, z_train_model)
        mse_ols_test[degree] = stat_tools.MSE(z_test, z_test_model)


        # CV, find best lambdas and get mse vs lambda for given degree.

        ridge_fold_score = np.zeros((n_lambdas, k_folds))
        lasso_fold_score = np.zeros((n_lambdas, k_folds))
        test_list, train_list = stat_tools.k_fold_selection(z, k_folds)
        for i in range(n_lambdas):
            lamb = lambdas[i]
            for j in range(k_folds):
                test_ind_cv = test_list[j]
                train_ind_cv = train_list[j]
                X_train_cv = X[train_ind_cv]
                z_train_cv = z[train_ind_cv]
                X_test_cv = X[test_ind_cv]
                z_test_cv = z[test_ind_cv]
                clf_Lasso = skl.Lasso(alpha=lamb,fit_intercept=False).fit(X_train_cv,z_train_cv)
                z_lasso_test = clf_Lasso.predict(X_test_cv)
                ridge_betas = linear_regression.Ridge_2D(X_train_cv, z_train_cv, lamb)
                z_ridge_test = X_test_cv @ ridge_betas
                ridge_fold_score[i,j] = stat_tools.MSE(z_test_cv, z_ridge_test)
                lasso_fold_score[i,j] = stat_tools.MSE(z_test_cv, z_lasso_test)

        lasso_cv_mse = np.mean(lasso_fold_score, axis=1)
        ridge_cv_mse = np.mean(ridge_fold_score, axis=1)
        best_lasso_lambda[degree] = lambdas[np.argmin(lasso_cv_mse)]
        best_ridge_lambda[degree] = lambdas[np.argmin(ridge_cv_mse)]
        best_lasso_mse[degree] = np.min(lasso_cv_mse)
        best_ridge_mse[degree] = np.min(ridge_cv_mse)
        lasso_lamb_deg_mse[degree] = lasso_cv_mse
        ridge_lamb_deg_mse[degree] = ridge_cv_mse


        # OLS bootstap, get bootstrapped mse, bias and variance for given degree.
        z_boot_model = np.zeros((len(z_test),n_bootstraps))
        for i in range(n_bootstraps):
            shuffle = np.random.randint(0,len(z_train),len(z_train))
            X_boot, z_boot = X_train_scaled[shuffle] , z_train[shuffle]
            betas_boot = linear_regression.OLS_SVD_2D(X_boot, z_boot)
            #betas_boot = linear_regression.Ridge_2D(X_boot, z_boot, lamb) #Ridge, given lambda
            #clf_Lasso = skl.Lasso(alpha=lamb).fit(X_boot,z_boot)
            #z_boot_model[:,i] = clf_Lasso_predict(X_test) #Lasso, given lambda
            z_boot_model[:,i] = X_test_scaled @ betas_boot
        mse, bias, variance = stat_tools.compute_mse_bias_variance(z_test, z_boot_model)
        ols_boot_mse[degree] = mse
        ols_boot_bias[degree] = bias
        ols_boot_variance[degree] = variance

        # Ridge bootstrap, get bootstrapped mse, bias and variance for given degree and lambda
        lamb = best_ridge_lambda[degree]
        z_boot_model = np.zeros((len(z_test),n_bootstraps))
        for i in range(n_bootstraps):
            shuffle = np.random.randint(0,len(z_train),len(z_train))
            X_boot, z_boot = X_train_scaled[shuffle] , z_train[shuffle]
            #betas_boot = linear_regression.OLS_SVD_2D(X_boot, z_boot)
            betas_boot = linear_regression.Ridge_2D(X_boot, z_boot, lamb) #Ridge, given lambda
            #clf_Lasso = skl.Lasso(alpha=lamb).fit(X_boot,z_boot)
            #z_boot_model[:,i] = clf_Lasso_predict(X_test) #Lasso, given lambda
            z_boot_model[:,i] = X_test_scaled @ betas_boot
        mse, bias, variance = stat_tools.compute_mse_bias_variance(z_test, z_boot_model)
        ridge_best_lambda_boot_mse[degree] = mse
        ridge_best_lambda_boot_bias[degree] = bias
        ridge_best_lambda_boot_variance[degree] = variance

        # Lasso bootstrap, get bootstrapped mse, bias and variance for given degree and lambda.
        lamb = best_lasso_lambda[degree]
        z_boot_model = np.zeros((len(z_test),n_bootstraps))
        for i in range(n_bootstraps):
            shuffle = np.random.randint(0,len(z_train),len(z_train))
            X_boot, z_boot = X_train_scaled[shuffle] , z_train[shuffle]
            #betas_boot = linear_regression.OLS_SVD_2D(X_boot, z_boot)
            #betas_boot = linear_regression.Ridge_2D(X_boot, z_boot, lamb) #Ridge, given lambda
            clf_Lasso = skl.Lasso(alpha=lamb,fit_intercept=False).fit(X_boot,z_boot)
            z_boot_model[:,i] = clf_Lasso.predict(X_test_scaled) #Lasso, given lambda
            #z_boot_model[:,i] = X_test_scaled @ betas_boot
        mse, bias, variance = stat_tools.compute_mse_bias_variance(z_test, z_boot_model)
        lasso_best_lambda_boot_mse[degree] = mse
        lasso_best_lambda_boot_bias[degree] = bias
        lasso_best_lambda_boot_variance[degree] = variance

################ All necessary computations should have been done above. Below follows
################ the plotting part.




        return



def deprecated_franke_analysis_full():
    n = 1000
    noise_scale = 0.2
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    z = FrankeFunction(x, y)
    # Adding standard normal noise:
    z = z + noise_scale*np.random.normal(0,1,len(z))
    max_degree = 20
    n_lambdas = 30
    n_bootstraps = 50
    k_folds = 5
    lambdas = np.logspace(-3,0,n_lambdas)
    subset_lambdas = lambdas[::5]

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
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2)
        # Scaling and feeding to CV.
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X[:,0] = 1

        # Scaling and feeding to bootstrap and OLS
        scaler_boot = StandardScaler()
        scaler_boot.fit(X_train)
        X_train_scaled = scaler_boot.transform(X_train)
        X_test_scaled = scaler_boot.transform(X_test)
        X_train_scaled[:,0] = 1
        X_test_scaled[:,0] = 1

        # OLS, get MSE for test and train set.

        betas = linear_regression.OLS_SVD_2D(X_train_scaled, z_train)
        z_test_model = X_test_scaled @ betas
        z_train_model = X_train_scaled @ betas
        mse_ols_train[degree] = stat_tools.MSE(z_train, z_train_model)
        mse_ols_test[degree] = stat_tools.MSE(z_test, z_test_model)


        # CV, find best lambdas and get mse vs lambda for given degree.

        ridge_fold_score = np.zeros((n_lambdas, k_folds))
        lasso_fold_score = np.zeros((n_lambdas, k_folds))
        test_list, train_list = stat_tools.k_fold_selection(z, k_folds)
        for i in range(n_lambdas):
            lamb = lambdas[i]
            for j in range(k_folds):
                test_ind_cv = test_list[j]
                train_ind_cv = train_list[j]
                X_train_cv = X[train_ind_cv]
                z_train_cv = z[train_ind_cv]
                X_test_cv = X[test_ind_cv]
                z_test_cv = z[test_ind_cv]
                clf_Lasso = skl.Lasso(alpha=lamb,fit_intercept=False).fit(X_train_cv,z_train_cv)
                z_lasso_test = clf_Lasso.predict(X_test_cv)
                ridge_betas = linear_regression.Ridge_2D(X_train_cv, z_train_cv, lamb)
                z_ridge_test = X_test_cv @ ridge_betas
                ridge_fold_score[i,j] = stat_tools.MSE(z_test_cv, z_ridge_test)
                lasso_fold_score[i,j] = stat_tools.MSE(z_test_cv, z_lasso_test)

        lasso_cv_mse = np.mean(lasso_fold_score, axis=1)
        ridge_cv_mse = np.mean(ridge_fold_score, axis=1)
        best_lasso_lambda[degree] = lambdas[np.argmin(lasso_cv_mse)]
        best_ridge_lambda[degree] = lambdas[np.argmin(ridge_cv_mse)]
        best_lasso_mse[degree] = np.min(lasso_cv_mse)
        best_ridge_mse[degree] = np.min(ridge_cv_mse)
        lasso_lamb_deg_mse[degree] = lasso_cv_mse
        ridge_lamb_deg_mse[degree] = ridge_cv_mse

        # All regressions bootstraps at once
        lamb_ridge = best_ridge_lambda[degree]
        lamb_lasso = best_lasso_lambda[degree]
        z_boot_ols = np.zeros((len(z_test),n_bootstraps))
        z_boot_ridge = np.zeros((len(z_test),n_bootstraps))
        z_boot_lasso= np.zeros((len(z_test),n_bootstraps))
        for i in range(n_bootstraps):
            shuffle = np.random.randint(0,len(z_train),len(z_train))
            X_boot, z_boot = X_train_scaled[shuffle] , z_train[shuffle]
            betas_boot_ols = linear_regression.OLS_SVD_2D(X_boot, z_boot)
            betas_boot_ridge = linear_regression.Ridge_2D(X_boot, z_boot, lamb_ridge) #Ridge, given lambda
            clf_Lasso = skl.Lasso(alpha=lamb_lasso,fit_intercept=False).fit(X_boot,z_boot)
            z_boot_lasso[:,i] = clf_Lasso.predict(X_test_scaled) #Lasso, given lambda
            z_boot_ridge[:,i] = X_test_scaled @ betas_boot_ridge
            z_boot_ols[:,i] = X_test_scaled @ betas_boot_ols

        ridge_best_lambda_boot_mse[degree], ridge_best_lambda_boot_bias[degree], \
        ridge_best_lambda_boot_variance[degree] = stat_tools.compute_mse_bias_variance(z_test, z_boot_ridge)

        lasso_best_lambda_boot_mse[degree], lasso_best_lambda_boot_bias[degree], \
        lasso_best_lambda_boot_variance[degree] = stat_tools.compute_mse_bias_variance(z_test, z_boot_lasso)

        ols_boot_mse[degree], ols_boot_bias[degree], \
        ols_boot_variance[degree] = stat_tools.compute_mse_bias_variance(z_test, z_boot_ols)

        # Bootstrapping for a selection of lambdas for ridge and lasso
        subset_lambda_index = 0
        for lamb in subset_lambdas:
            z_boot_ridge = np.zeros((len(z_test),n_bootstraps))
            z_boot_lasso= np.zeros((len(z_test),n_bootstraps))
            for i in range(n_bootstraps):
                shuffle = np.random.randint(0,len(z_train),len(z_train))
                X_boot, z_boot = X_train_scaled[shuffle] , z_train[shuffle]
                betas_boot_ridge = linear_regression.Ridge_2D(X_boot, z_boot, lamb) #Ridge, given lambda
                clf_Lasso = skl.Lasso(alpha=lamb,fit_intercept=False).fit(X_boot,z_boot)
                z_boot_lasso[:,i] = clf_Lasso.predict(X_test_scaled) #Lasso, given lambda
                z_boot_ridge[:,i] = X_test_scaled @ betas_boot_ridge

            ridge_subset_lambda_boot_mse[degree, subset_lambda_index ], ridge_subset_lambda_boot_bias[degree, subset_lambda_index ], \
            ridge_subset_lambda_boot_variance[degree, subset_lambda_index ] = stat_tools.compute_mse_bias_variance(z_test, z_boot_ridge)

            lasso_subset_lambda_boot_mse[degree, subset_lambda_index ], lasso_subset_lambda_boot_bias[degree, subset_lambda_index ], \
            lasso_subset_lambda_boot_variance[degree, subset_lambda_index ] = stat_tools.compute_mse_bias_variance(z_test, z_boot_lasso)

            subset_lambda_index  += 1
