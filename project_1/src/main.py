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


def R2(y_data, y_model):
    # Computes the confidence number
    return 1 - np.sum((y_data - y_model)**2) / np.sum((y_data - np.mean(y_data))**2)


def MSE(y_data, y_model):
    # Computes the mean squared error
    return np.sum((y_data - y_model)**2) / np.size(y_model)

def var_beta(y_data, X):
    """ Computes the covariance matrix
    Args:
        y_data (Array): Data points.
        X (Array): Design matrix corresponding to y_data

    Returns:
        Array: Covariance Matrix
    """
    return np.sqrt(np.var(y_data) * np.linalg.inv(X.T @ X).diagonal())


def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def part_1a():
    # Sample the franke function n times at randomly chosen points
    n = 100
    deg = 5
    noise_scale = 0.2
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    z = FrankeFunction(x, y)
    # Adding standard normal noise:
    z_noisy = z + noise_scale*np.random.normal(0,1,len(z))
    # Making the design matrix
    X = linear_regression.design_matrix_2D(x,y,deg)
    # Find the least-squares solution
    beta = linear_regression.OLS_2D(X, z)
    beta_noisy = linear_regression.OLS_2D(X, z_noisy)

    # Split into training and test data with ratio 0.2
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2)
    # Scale data according to sklearn, beware possible problems with intercept and std.
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # For ridge and lasso, lasso directly from sklearn.
    # For given polynomial degree, input X and z. X should be prescaled.

    n_lambdas = 100
    lambdas = np.logspace(-3,0,n_lambdas)
    k_folds = 5
    ridge_fold_score = np.zeros(n_lambda, k_folds)
    lasso_fold_score = np.zeros(n_lambda, k_folds)
    test_list, train_list = k_fold_selection(z, k_folds)
    for i in range(n_lambdas):

        for j in range(k_folds):
            test_ind_cv = test_list[j]
            train_ind_cv = train_list[j]
            X_train_cv = X[train_ind_cv]
            z_train_cv = z[train_ind_cv]
            X_test_cv = X[test_ind_cv]
            z_test_cv = z[test_ind_cv]
            clf_Lasso = skl.Lasso(alpha=lamb).fit(X_train_cv,z_train_cv)
            z_lasso_test = clf_Lasso.predict(X_test_cv)
            ridge_betas = linear_regression.Ridge_2D(X_train_cv, z_train_cv, lamb)
            z_ridge_test = X_test_cv @ ridge_betas
            ridge_fold_score[i,j] = stat_tools.MSE(z,z_ridge_test)
            lasso_fold_score[i,j] = stat_tools.MSE(z,z_lasso_test)

    lasso_cv_mse = np.mean(lasso_fold_score, axis=1, keepdims=True)
    ridge_cv_mse = np.mean(ridge_fold_score, axis=1, keepdims =True)
    best_lambda_lasso = lambdas[np.argmin(lasso_cv_mse)]
    best_lambda_ridge = lambdas[np.argmin(ridge_cv_mse)]



    # Bootstrap skeleton
    # For given polynomial degree, input X_train, z_train, X_test and z_test.
    # X_train and X_test should be scaled?
    n_bootstraps = 100
    z_boot_model = np.zeros(len(z_test),n_bootstraps)
    for bootstrap_number in range(n_bootstraps):
        # For the number of data value points (len_z) in the training set, pick a random
        # data value (z_train[random]) and its corresponding row in the design matrix
        shuffle = np.random.randint(0,len(z_train),len(z_train))
        X_boot, z_boot = X_train[shuffle] , z_train[shuffle]
        betas_boot = linear_regression.OLS_SVD_2D(X_boot, z_boot)
        #betas_boot = linear_regression.Ridge_2D(X_boot, z_boot, lamb) #Ridge, given lambda
        #clf_Lasso = skl.Lasso(alpha=lamb).fit(X_boot,z_boot)
        #z_boot_model[:,i] = clf_Lasso_predict(X_test) #Lasso, given lambda
        z_boot_model[:,i] = X_test @ betas_boot
    mse, bias, variance = stat_tools.compute_mse_bias_variance(z_test, z_boot_model)




    # Check MSE
    print("MSE = %.3f" % MSE(z, linear_regression.evaluate_poly_2D(x, y, beta, deg)))
    # And with noise
    print("Including standard normal noise scaled by {}, MSE = {:.3f}".format(
        noise_scale, MSE(z_noisy, linear_regression.evaluate_poly_2D(x, y, beta_noisy, deg))))
    # Evaluate the Franke function & least-squares
    x = np.linspace(0, 1, 30)
    y = np.linspace(0, 1, 30)
    X, Y = np.meshgrid(x, y)

    z_analytic = FrankeFunction(X, Y)
    z_fit = linear_regression.evaluate_poly_2D(X, Y, beta, deg)
    z_fit_noisy = linear_regression.evaluate_poly_2D(X, Y, beta_noisy, deg)

    fig = plt.figure()

    # Plot the analytic curve
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.set_title("Franke Function")
    ax.view_init(azim=45)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    surf = ax.plot_surface(X, Y, z_analytic, cmap=cm.coolwarm)

    # Plot the fitted curve
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.set_title("OLS")
    ax.view_init(azim=45)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    surf = ax.plot_surface(X, Y, z_fit, cmap=cm.coolwarm)

    # Plot fitted curve, with noisy beta estimates
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    ax.set_title("OLS with noise")
    ax.view_init(azim=45)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    surf = ax.plot_surface(X, Y, z_fit_noisy, cmap=cm.coolwarm)

    plt.show()

    return

def franke_analysis():
    # Structure what we are doing

    return

def terrain_analysis():
    # Setting up the terrain data:
    terrain_data = imread('path_to_data')
    x_terrain = np.arange(terrain_data.shape[1]) #apparently, from the problem description.
    y_terrain = np.arange(terrain_data.shape[0])
    X, Y = np.meshgrid(x_terrain,y_terrain)
    z_terrain = terrain_data.flatten() # the response values
    x_terrain_flat = X.flatten() # the first degree feature variables
    y_terrain_flat = Y.flatten() # the first degree feature variables
    max_degree = 10
    n_lambdas = 100
    n_bootstraps = 100
    k_folds = 5
    lambdas = np.logspace(-3,0,n_lambdas)

    # Quantities of interest:
    mse_ols_test = np.zeros(max_degree)
    mse_ols_train = np.zeros(max_degree)
    ols_boot_bias = np.zeros(max_degree)
    ols_boot_variance = np.zeros(max_degree)
    best_ridge_mse = np.zeros(max_degree)
    ridge_best_lambda_boot_bias = np.zeros(max_degree)
    ridge_best_lambda_boot_variance = np.zeros(max_degree)
    best_lasso_mse = np.zeros(max_degree)
    lasso_best_lambda_boot_bias = np.zeros(max_degree)
    lasso_best_lambda_boot_variance = np.zeros(max_degree)
    ridge_lamb_deg_mse = np.zeros(max_degree, n_lambdas)
    lasso_lamb_deg_mse = np.zeros(max_degree, n_lambdas)


    # Actual computations
    for degree in range(max_degree):
        X_terrain_design = linear_regression.design_matrix_2D(x,y,degree)

        return

if __name__ == "__main__":
    part_1a()
    franke_analysis()
    terrain_analysis()
