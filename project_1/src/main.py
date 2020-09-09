import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import OLS
import linear_regression
import utils


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
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    z = FrankeFunction(x, y)
    # Find the least-squares solution
    beta = OLS.OLS_2D(x, y, z, n=deg)

    # Check MSE
    print("MSE = %.3f" % MSE(FrankeFunction(x, y), linear_regression.evaluate_poly_2D(x, y, beta, deg)))

    # Evaluate the Franke function & least-squares
    x = np.linspace(0, 1, 30)
    y = np.linspace(0, 1, 30)
    X, Y = np.meshgrid(x, y)

    z_analytic = FrankeFunction(X, Y)
    z_fit = linear_regression.evaluate_poly_2D(X, Y, beta, deg)

    fig = plt.figure()

    # Plot the analytic curve
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.set_title("Franke Function")
    ax.view_init(azim=45)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    surf = ax.plot_surface(X, Y, z_analytic, cmap=cm.coolwarm)

    # Plot the fitted curve
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.set_title("OLS")
    ax.view_init(azim=45)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    surf = ax.plot_surface(X, Y, z_fit, cmap=cm.coolwarm)

    plt.show()

    return


if __name__ == "__main__":
    part_1a()
