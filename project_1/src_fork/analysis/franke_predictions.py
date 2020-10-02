import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as skl
from imageio import imread

import sys
sys.path.insert(0, "../")

import linear_regression
import utils
import stat_tools

def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


def franke_predictions(
    n=1000, noise_scale=0.2, degree=20, ridge_lambda=1e-2, lasso_lambda=1e-5, plot_grid_size=2000
):
    """For a given sample size n, noise_scale, max_degree and penalty parameters: produces ols,
    ridge and lasso predictions, as well as ground truth on a plotting meshgrid with input grid size.

    output:
        x_plot_mesh: meshgrid of x-coordinates
        y_plot_mesh: meshgrid of y-coordinates
        z_predict_ols: ols prediction of z on the meshgrid
        z_predict_ridge: ridge prediction of z on the meshgrid
        z_predict_lasso: lasso prediction of z on the meshgrid
        z_plot_franke: Actual Franke values on the meshgrid.

    """

    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    z = FrankeFunction(x, y)
    # Adding standard normal noise:
    z = z + noise_scale * np.random.normal(0, 1, len(z))
    #   Centering the response
    z_intercept = np.mean(z)
    z = z - z_intercept
    # Scaling
    X = linear_regression.design_matrix_2D(x, y, degree)
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    # Setting up plotting grid
    x_plot = np.linspace(0, 1, plot_grid_size)
    y_plot = np.linspace(0, 1, plot_grid_size)
    x_plot_mesh, y_plot_mesh = np.meshgrid(x_plot, y_plot)
    x_plot_mesh_flat, y_plot_mesh_flat = x_plot_mesh.flatten(), y_plot_mesh.flatten()

    X_plot_design = linear_regression.design_matrix_2D(x_plot_mesh_flat, y_plot_mesh_flat, degree)
    X_plot_design_scaled = scaler.transform(X_plot_design)

    z_plot_franke = FrankeFunction(x_plot_mesh, y_plot_mesh)

    # OLS
    betas = linear_regression.OLS_SVD_2D(X_scaled, z)
    z_predict_flat_ols = (X_plot_design_scaled @ betas) + z_intercept
    z_predict_ols = z_predict_flat_ols.reshape(plot_grid_size, -1)

    # Ridge

    betas_ridge = linear_regression.Ridge_2D(X_scaled, z, ridge_lambda)
    z_predict_flat_ridge = (X_plot_design_scaled @ betas_ridge) + z_intercept
    z_predict_ridge = z_predict_flat_ridge.reshape(plot_grid_size, -1)
    # Lasso

    clf_Lasso = skl.Lasso(alpha=lasso_lambda, fit_intercept=False, max_iter=10000).fit(X_scaled, z)
    z_predict_flat_lasso = clf_Lasso.predict(X_plot_design_scaled) + z_intercept
    z_predict_lasso = z_predict_flat_lasso.reshape(plot_grid_size, -1)

    return x_plot_mesh, y_plot_mesh, z_predict_ols, z_predict_ridge, z_predict_lasso, z_plot_franke


if __name__ == "__main__":
    (
        x_plot_mesh,
        y_plot_mesh,
        z_predict_ols,
        z_predict_ridge,
        z_predict_lasso,
        z_plot_franke,
    ) = franke_predictions()

    fig = plt.figure()

    # Plot the analytic curve
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    ax.set_title("Franke Function")
    ax.view_init(azim=45)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    surf = ax.plot_surface(x_plot_mesh, y_plot_mesh, z_plot_franke, cmap=cm.coolwarm)

    # Plot the OLS prediction
    ax = fig.add_subplot(1, 4, 2, projection="3d")
    ax.set_title("OLS")
    ax.view_init(azim=45)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    surf = ax.plot_surface(x_plot_mesh, y_plot_mesh, z_predict_ols, cmap=cm.coolwarm)

    ax = fig.add_subplot(1, 4, 3, projection="3d")
    ax.set_title("Ridge")
    ax.view_init(azim=45)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    surf = ax.plot_surface(x_plot_mesh, y_plot_mesh, z_predict_ridge, cmap=cm.coolwarm)

    ax = fig.add_subplot(1, 4, 4, projection="3d")
    ax.set_title("Lasso")
    ax.view_init(azim=45)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    surf = ax.plot_surface(x_plot_mesh, y_plot_mesh, z_predict_lasso, cmap=cm.coolwarm)

    plt.show()
