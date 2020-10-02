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

def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def franke_analysis_plots(n=1000,noise_scale=0.2,max_degree=20,n_bootstraps=100,k_folds=5,n_lambdas=30, do_boot=True, do_subset=True):


# Note that max_degrees is the number of degrees, i.e. including 0.

    # n = 500
    # noise_scale = 0.2
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    z = FrankeFunction(x, y)
    # Adding standard normal noise:
    z = z + noise_scale*np.random.normal(0,1,len(z))
    # max_degree = 15
    # n_lambdas = 30
    # n_bootstraps = 100
    # k_folds = 5
    lambdas = np.logspace(-5,0,n_lambdas)
    subset_lambdas = lambdas[::10]

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

    # CV MSE for OLS:
    plt.figure()
    plt.semilogy(ols_cv_mse)
    plt.title('OLS CV MSE')
    plt.show()

    # Bootstrap for OLS:
    plt.figure()
    plt.semilogy(ols_boot_mse,label='mse')
    plt.semilogy(ols_boot_bias,label='bias')
    plt.semilogy(ols_boot_variance, label='variance')
    plt.title('OLS bias-variance-MSE by bootstrap')
    plt.legend()
    plt.show()

    # CV for Ridge, best+low+middle+high lambdas
    plt.figure()
    plt.semilogy(best_ridge_mse, label='best for each degree')
    plt.semilogy(ridge_lamb_deg_mse[:,0], label='lambda={}'.format(lambdas[0]))
    plt.semilogy(ridge_lamb_deg_mse[:,10], label='lambda={}'.format(lambdas[10]))
    plt.semilogy(ridge_lamb_deg_mse[:,20], label='lambda={}'.format(lambdas[20]))
    plt.title('Ridge CV MSE for best lambda at each degree, plus for given lambdas across all degrees')
    plt.legend()
    plt.show()

    # Bootstrap for the best ridge lambdas:
    plt.figure()
    plt.semilogy(ridge_best_lambda_boot_mse, label='mse')
    plt.semilogy(ridge_best_lambda_boot_bias, label ='bias')
    plt.semilogy(ridge_best_lambda_boot_variance, label='variance')
    plt.title('Best ridge lambdas for each degree bootstrap')
    plt.legend()
    plt.show()

    # Bootstrap only bias and variance for low+middle+high ridge lambdas

    plt.figure()
    plt.semilogy(ridge_subset_lambda_boot_bias[0], label = 'bias, lamdbda = {}'.format(subset_lambdas[0]))
    plt.semilogy(ridge_subset_lambda_boot_variance[0], label = 'variance, lamdbda = {}'.format(subset_lambdas[0]))
    plt.semilogy(ridge_subset_lambda_boot_bias[1],label = 'bias, lamdbda = {}'.format(subset_lambdas[1]))
    plt.semilogy(ridge_subset_lambda_boot_variance[1],label = 'variance, lamdbda = {}'.format(subset_lambdas[1]))
    plt.semilogy(ridge_subset_lambda_boot_bias[2],label = 'bias, lamdbda = {}'.format(subset_lambdas[2]))
    plt.semilogy(ridge_subset_lambda_boot_variance[2],label = 'variance, lamdbda = {}'.format(subset_lambdas[2]))
    plt.title('Bias+variance for low, middle, high ridge lambdas')
    plt.legend()
    plt.show()

    # CV for lasso, best+low+middle+high lambdas
    plt.figure()
    plt.semilogy(best_lasso_mse,legend='best lambda for each degree')
    plt.semilogy(lasso_lamb_deg_mse[:,0],label='lambda={}'.format(lambdas[0]))
    plt.semilogy(lasso_lamb_deg_mse[:,10],label='lambda={}'.format(lambdas[10]))
    plt.semilogy(lasso_lamb_deg_mse[:,20],label='lambda={}'.format(lambdas[20]))
    plt.title('Lasso CV MSE for best lambda at each degree, plus for given lambdas across all degrees')
    plt.legend()
    plt.show()

    # Bootstrap for the best lasso lambdas:
    plt.figure()
    plt.semilogy(lasso_best_lambda_boot_mse, label='mse')
    plt.semilogy(lasso_best_lambda_boot_bias, label='bias')
    plt.semilogy(lasso_best_lambda_boot_variance, label='variance')
    plt.title('Best lasso lambdas for each degree bootstrap')
    plt.legend()
    plt.show()

    # Bootstrap only bias and variance for low+middle+high lasso lambdas

    plt.figure()
    plt.semilogy(lasso_subset_lambda_boot_bias[0],label = 'bias, lamdbda = {}'.format(subset_lambdas[0]))
    plt.semilogy(lasso_subset_lambda_boot_variance[0],label = 'variance, lamdbda = {}'.format(subset_lambdas[0]))
    plt.semilogy(lasso_subset_lambda_boot_bias[1],label = 'bias, lamdbda = {}'.format(subset_lambdas[1]))
    plt.semilogy(lasso_subset_lambda_boot_variance[1],label = 'variance, lamdbda = {}'.format(subset_lambdas[1]))
    plt.semilogy(lasso_subset_lambda_boot_bias[2],label = 'bias, lamdbda = {}'.format(subset_lambdas[2]))
    plt.semilogy(lasso_subset_lambda_boot_variance[2],label = 'variance, lamdbda = {}'.format(subset_lambdas[2]))
    plt.title('Bias+variance for low, middle, high lasso lambdas')
    plt.legend()
    plt.show()

    # For a couple of degrees, plot cv mse vs lambda for ridge, will break program if max_degrees < 8

    plt.figure()
    plt.plot(np.log10(lambdas), ridge_lamb_deg_mse[max_degree-1], label = 'degree = {}'.format(max_degree-1))
    plt.plot(np.log10(lambdas), ridge_lamb_deg_mse[max_degree-2], label = 'degree = {}'.format(max_degree-2))
    plt.plot(np.log10(lambdas), ridge_lamb_deg_mse[max_degree-3], label = 'degree = {}'.format(max_degree-3))
    plt.plot(np.log10(lambdas), ridge_lamb_deg_mse[max_degree-5], label = 'degree = {}'.format(max_degree-5))
    plt.plot(np.log10(lambdas), ridge_lamb_deg_mse[max_degree-7], label = 'degree = {}'.format(max_degree-7))
    plt.legend()
    plt.show()

    # For a copule of degrees, plot cv mse vs lambda for lasso, will break program if max_degree < 8.

    plt.figure()
    plt.plot(np.log10(lambdas), lasso_lamb_deg_mse[max_degree-1], label = 'degree = {}'.format(max_degree-1))
    plt.plot(np.log10(lambdas), lasso_lamb_deg_mse[max_degree-2], label = 'degree = {}'.format(max_degree-2))
    plt.plot(np.log10(lambdas), lasso_lamb_deg_mse[max_degree-3], label = 'degree = {}'.format(max_degree-3))
    plt.plot(np.log10(lambdas), lasso_lamb_deg_mse[max_degree-5], label = 'degree = {}'.format(max_degree-5))
    plt.plot(np.log10(lambdas), lasso_lamb_deg_mse[max_degree-7], label = 'degree = {}'.format(max_degree-7))
    plt.legend()
    plt.show()

    return


if __name__ == '__main__':
    franke_analysis_plots()
