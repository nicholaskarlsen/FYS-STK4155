import numpy as np
import matplotlib.pyplot as plt

# import SGD
import ActivationFunctions
import CostFunctions
import SGD
import sys

sys.path.insert(0, "../../project_1/src")

from stat_tools import MSE, R2


def logreg(x, y, M, init_w, n_epochs, learning_rate, momentum, lambd=None):
    """Logistic (softmax) regression for multiclass categorization. Uses momentum SGD."""

    # prob = np.exp(X @ theta)/np.sum(np.exp(X @ theta),axis=1)
    # prob.shape = [examples,classes]. Category probabilites as given
    # by the Softmax-function.

    # cost_function = -np.sum(Y * np.ln(prob) + (1-Y)*np.ln(prob))
    # scalar, full cross-entropy cost function. Note that the last term
    # is not really necessary, when using softmax there is an implicit
    # penalty for giving nonzero probabilites to false categories.
    # This version is a little more aggressive in punishing confidence in wrong
    # labels, but is far clunkier and possibly numerically unstable.

    # cost_function = -np.sum(Y * np.ln(prob))
    # is the preferred cost function when using softmax. Also given as
    # Softmax_loss in CostFunctions.py

    # shorthand = prob * (Y / prob - (1-Y)/(1-prob))
    # helper matrix for vectorizing the computation of the cost_gradients
    # shape = [examples,classes]. Remove the last term if using Softmax_loss
    # as cost_function.

    # cost_gradients = X.T @ (prob * np.sum(shorthand,axis=1)) - X.T @ shorthand
    # Derivatives of cost wrt thetas, shape = [predictors,classes]

    # NOTE! The above expressions for the cost_gradients have not been
    # double-checked! Given the potential for error in their derivations
    # they should be treated as highly suspect. The simpler version for
    # Softmax_loss implemented below, is however, pretty safe.

    w = init_w
    dw = np.zeros(w.shape)

    for epoch in range(n_epochs):
        mb = SGD.minibatch(x, M)  # Split x into M minibatches
        for i in range(M):
            # Pick out a random mini-batch index
            k = np.random.randint(M)
            # compute gradient with random minibatch
            X, Y = x[mb[k]], y[mb[k]]
            # Probabilities from Softmax:
            prob = np.exp(X @ w) / np.sum(np.exp(X @ w), axis=1, keepdims=True)

            # cost_function = -np.sum(Y * np.ln(prob)) gives:
            grad = X.T @ (prob - Y)

            # Add l2 penalty:
            if lambd != None:
                grad += 2 * lambd * w

            # increment weights
            dw = momentum * dw - learning_rate * grad / X.shape[0]
            w = w + dw
    return w
