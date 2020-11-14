import numpy as np
import sys

import NeuralNetwork as NN
import CostFunctions
import SGD

sys.path.insert(0, "../../project_1/src")
from crossvalidation import k_fold_selection
from stat_tools import MSE


def CrossValidation(X, y, score_func, k_folds):
    """Performs k-fold Cross validation of some data set X, y for some generic
    score_func which is assumed to contain all the setting up and logics of solving
    the problem, simply to return an MSE/R2 score.
    """
    # Split into 5 sets of training/testing indices
    test_indices, train_indices = k_fold_selection(X, k_folds)

    score = 0
    for i in range(k_folds):
        # Fetch out the relevant indices
        train_indices_fold = train_indices[i]
        test_indices_fold = test_indices[i]
        # Fetch out the relevant data
        X_training = X[train_indices_fold]
        y_training = y[train_indices_fold]
        X_test = X[test_indices_fold]
        y_test = y[test_indices_fold]
        # Compute the MSE or other relevant performance metric
        score += score_func(X_training, y_training, X_test, y_test)
    # Average the metric over all the folds
    score /= k_folds

    return score


"""
Functions which generates functions that score their respective methods for usage with the
CrossValidation function above
"""


def generate_SGDM_scorefunc(M, init_w, n_epochs, learning_rate, momentum, cost_gradient, *lambd):
    def SGDM_score_func(X_training, y_training, X_test, y_test):
        weights = SGD.SGDM(
            x=X_training,
            y=y_training,
            M=M,
            init_w=init_w,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            momentum=momentum,
            cost_gradient=cost_gradient,
            lambd=lambd,
        )

        return stat_tools.MSE(y_test, X_test @ weights)

    return SGDM_score_func


def generate_FFNN_score_func(
    N_minibatches,
    learning_rate,
    n_epochs,
    network_shape,
    activation,
    activation_out,
    cost=CostFunctions.SquareError,
    momentum=0,
    lambd=None,
    init_weights_method=None,
):
    """
    Wrapper function which perform all the required setup for using the general CV scheme by
    generating a score function.
    """

    def FFNNScoreFunc(X_training, y_training, X_test, y_test):
        # Initialize the neural  network based
        FFNN = NN.FeedForwardNeuralNetwork(
            X=X_training,
            Y=y_training,
            network_shape=network_shape,
            activation=activation_out,
            activation_out=activation_out,
            cost=cost,
            momentum=momentum,
            lambd=lambd,
            init_weights_method=init_weights_method,
        )

        FFNN.train(N_minibatches=N_minibatches, learning_rate=learning_rate, n_epochs=n_epochs)

        return FFNN.score(y_test=y_test, X_test=X_test)

    return FFNNScoreFunc


def generate_FFNNClassifier_score_func(
    N_minibatches,
    learning_rate,
    n_epochs,
    network_shape,
    activation,
    activation_out,
    momentum=0,
    lambd=None,
    init_weights_method=None,
):
    """
    Wrapper function which perform all the required setup for using the general CV scheme by
    generating a score function.
    """

    def FFNNClassifierScoreFunc(X_training, y_training, X_test, y_test):
        # Initialize the neural  network based
        FFNN = NN.FeedForwardNeuralNetwork(
            X=X_training,
            Y=y_training,
            network_shape=network_shape,
            activation=activation_out,
            activation_out=activation_out,
            momentum=momentum,
            lambd=lambd,
            init_weights_method=init_weights_method,
        )

        FFNN.train(N_minibatches=N_minibatches, learning_rate=learning_rate, n_epochs=n_epochs)

        return FFNN.score(y_test=y_test, X_test=X_test)

    return FFNNScoreFunc
