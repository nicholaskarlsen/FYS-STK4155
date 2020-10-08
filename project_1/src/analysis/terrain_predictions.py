import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imageio import imread

import sys

sys.path.insert(0, "../")
import linear_regression
import utils
import stat_tools
import crossvalidation
import bootstrap

utils.plot_settings()  # LaTeX fonts in Plots!


def terrain_predictions(spacing=40, degree=20, ridge_lambda=1e-2, lasso_lambda=1e-2):
    """For a given sampling spacing, degree and penalty parameters: produces ols,
    ridge and lasso predictions, as well as ground truth on a plotting meshgrid.

    output:
        x_plot_mesh: meshgrid of x-coordinates
        y_plot_mesh: meshgrid of y-coordinates
        z_predict_ols: ols prediction of z on the meshgrid
        z_predict_ridge: ridge prediction of z on the meshgrid
        z_predict_lasso: lasso prediction of z on the meshgrid
        z_true: Actual terrain values on the meshgrid.

    """
    # #control variables, resticted to upper half of plot currently.
    # spacing = 10
    # degree = 25
    # ridge_lambda = 1e-2
    # lasso_lambda = 1e-5
    np.random.seed(2018)
    # Setting up the terrain data:
    # Note structure! X-coordinates are on the rows of terrain_data
    # Point_selection.flatten() moves most rapidly over the x-coordinates
    # Meshgrids flattened also move most rapidly over the x-coordinates. Thus
    # this should make z(x,y).reshape(length_y,length_x) be consistent with terrain_data
    terrain_data = imread("../../datafiles/SRTM_data_Norway_1.tif")
    point_selection = terrain_data[:1801:spacing, :1801:spacing]  # Make quadratic and downsample
    x_terrain_selection = np.linspace(0, 1, point_selection.shape[1])
    y_terrain_selection = np.linspace(0, 1, point_selection.shape[0])
    X_coord_selection, Y_coord_selection = np.meshgrid(x_terrain_selection, y_terrain_selection)
    z_terrain_selection = point_selection.flatten()  # the response values
    x_terrain_selection_flat = X_coord_selection.flatten()  # the first degree feature variables
    y_terrain_selection_flat = Y_coord_selection.flatten()  # the first degree feature variables

    x = x_terrain_selection_flat
    y = y_terrain_selection_flat
    z = z_terrain_selection

    # Centering
    z_intercept = np.mean(z)
    z = z - z_intercept
    # Scaling
    X = linear_regression.design_matrix_2D(x, y, degree)
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    X_scaled = X_scaled[:,1:]

    x_plot = np.linspace(0, 1, 1801)
    y_plot = np.linspace(0, 1, 1801)
    x_plot_mesh, y_plot_mesh = np.meshgrid(x_plot, y_plot)
    x_plot_mesh_flat, y_plot_mesh_flat = x_plot_mesh.flatten(), y_plot_mesh.flatten()

    X_plot_design = linear_regression.design_matrix_2D(x_plot_mesh_flat, y_plot_mesh_flat, degree)
    X_plot_design_scaled = scaler.transform(X_plot_design)
    X_plot_design_scaled = X_plot_design_scaled[:,1:]
    # Ground truth

    z_true = terrain_data[:1801, :1801]

    # OLS
    betas = linear_regression.OLS_SVD_2D(X_scaled, z)
    z_predict_flat_ols = (X_plot_design_scaled @ betas) + z_intercept
    z_predict_ols = z_predict_flat_ols.reshape(-1, 1801)

    # Ridge
    betas_ridge = linear_regression.Ridge_2D(X_scaled, z, ridge_lambda)
    z_predict_flat_ridge = (X_plot_design_scaled @ betas_ridge) + z_intercept
    z_predict_ridge = z_predict_flat_ridge.reshape(-1, 1801)
    # Lasso

    clf_Lasso = skl.Lasso(alpha=lasso_lambda, fit_intercept=False).fit(X_scaled, z)
    z_predict_flat_lasso = clf_Lasso.predict(X_plot_design_scaled) + z_intercept
    z_predict_lasso = z_predict_flat_lasso.reshape(-1, 1801)

    return x_plot_mesh, y_plot_mesh, z_predict_ols, z_predict_ridge, z_predict_lasso, z_true


if __name__ == "__main__":
    (
        x_plot_mesh,
        y_plot_mesh,
        z_predict_ols,
        z_predict_ridge,
        z_predict_lasso,
        z_truth,
    ) = terrain_predictions()

    fig = plt.figure()

    # Plot the true terrain data
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    ax.set_title("True terrain data")
    ax.view_init(azim=45, elev=60)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    surf = ax.plot_surface(x_plot_mesh, y_plot_mesh, z_truth, cmap=cm.coolwarm)

    # Plot the OLS prediction
    ax = fig.add_subplot(1, 4, 2, projection="3d")
    ax.set_title("OLS")
    ax.view_init(azim=45, elev=60)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    surf = ax.plot_surface(x_plot_mesh, y_plot_mesh, z_predict_ols, cmap=cm.coolwarm)

    ax = fig.add_subplot(1, 4, 3, projection="3d")
    ax.set_title("Ridge")
    ax.view_init(azim=45, elev=60)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    surf = ax.plot_surface(x_plot_mesh, y_plot_mesh, z_predict_ridge, cmap=cm.coolwarm)

    ax = fig.add_subplot(1, 4, 4, projection="3d")
    ax.set_title("Lasso")
    ax.view_init(azim=45, elev=60)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    surf = ax.plot_surface(x_plot_mesh, y_plot_mesh, z_predict_lasso, cmap=cm.coolwarm)

    plt.show()
