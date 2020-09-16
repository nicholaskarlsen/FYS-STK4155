import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import linear_regression
import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as skl

utils.plot_settings() # LaTeX fonts in Plots!


def R2(y_data, y_model):
    # Computes the confidence number
    return 1 - np.sum((y_data - y_model)**2) / np.sum((y_data - np.mean(y_data))**2)


def MSE(y_data, y_model):
    # Computes the mean squared error
    return np.sum((y_data - y_model)**2) / np.size(y_model)


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
    n_lambdas = 100
    lambdas = np.logspace(-3,0,n_lambdas)
    for lamb in lamdas:
        clf_Lasso = skl.Lasso(alpha=lamb).fit(X_train,z_train)
        y_Lasso = clf_Lasso.predict(X_test)

    # Bootstrap skeleton

    n_bootstraps = 100
    for bootstrap_number in range(n_bootstraps):
        # For the number of data value points (len_z) in the training set, pick a random
        # data value (z_train[random]) and its corresponding row in the design matrix
        shuffle = np.random.randint(0,len(z_train),len(z_train))
        X_boot, z_boot = X_train[shuffle] , z_train[shuffle]
        # Do whatever


    # k-fold CV skeleton
    k_folds = 5
    fold_number = np.random.randint(0,k_folds,len(z_train))
    for k in range(k_folds):
        test_index = np.where(fold_number = k)
        z_folded_test = z_train[test_index]
        x_folded_test = X_train[test_index]
        x_folded = X_train[np.logical_not(test_index)]
        z_folded = z_train[np.logical_not(test_index)]

        #do whatever

    #Alternatively for k-fold

    k_folds = 5
    elements_per_bin = int(len(z_train)/k_folds)
    permutations = np.random.permutation(np.arange(len(z_train)))
    for k in range(k_folds):
        test_mask = np.ones(len(z_train), bool)
        test_mask[k*elements_per_bin:(k+1)*elements_per_bin] = False
        z_folded_test = z_train[permutations[np.logical_not(test_mask)]]
        z_folded_train = z_train[permutations[test_mask]]

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


if __name__ == "__main__":
    part_1a()
