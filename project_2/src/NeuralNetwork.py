import numpy as np
import matplotlib.pyplot as plt

import SGD
import ActivationFunctions
import CostFunctions
import sys


class FeedForwardNeuralNetwork:

    def __init__(self, X, y, network_shape, activation, activation_out, cost, *lambd):
        """ Models implements a Feed Forward Neural Network
        Args:
            network_shape (Array) : Defines the number of hidden layers (L) and number of neurons
                    (n) for each hidden layer in the network in the form [n_1, n_2, ..., n_L].

            activation (Object)   : The activation function; MUST be an implementation of the
                    ActivationFunction interface. See ActivationFunctions.py

            cost (Object)         : The cost function. Same as above; must be implementation of
                    CostFunction interface.
        """
        self.X = X
        self.y = y

        try:
            self.N_inputs, self.input_dim = self.X.shape
        except ValueError as e: # if 1D input data
            self.N_inputs = len(self.X)
            self.input_dim = 1

        try:
            self.N_outputs, self.output_dim = self.y.shape
        except ValueError as e: # if 1D output data
            self.N_outputs = len(self.y)
            self.output_dim = 1

        # Make sure both data-sets are the same size
        assert self.N_inputs == self.N_outputs

        # Ensure that activation & cost are implementations of their respective interfaces
        # Which in turn guaranties the existence of the appropriate (static) methods
        if issubclass(activation, ActivationFunctions.ActivationFunction):
            self.activation = activation
        else:
            raise Exception(
                "activation is not a sub-class of ActivationFunction")

        if issubclass(activation_out, ActivationFunction):
            self.activation_out = activation
        else:
            raise Exception(
                "activation_out is not a sub-class of ActivationFunction")

        if issubclass(cost, CostFunctions.CostFunction):
            self.cost = cost
        else:
            raise Exception(
                "cost is not a sub-class of CostFunctions")

        # Initialize weight & bias
        self.network_shape = network_shape
        self.N_layers = len(self.network_shape)
        self.weights = np.empty(self.N_layers + 2, dtype='object')
        self.biases = np.empty(self.N_layers + 2, dtype='object')
        self.__initialize_weights()
        self.__initialize_biases()
        return

    def __initialize_weights(self):
        # NOTE: Consult the literature to ensure that random initialization is OK
        # weight from k in l-1 to j in l -> w[l][j,k]
        self.weights[0] = np.random.randn()
        for L, N_w in enumerate(self.network_shape):
            self.weights[L+1] = np.random.randn(self.network_shape[L])

        return

    def __initialize_biases(self):
        # NOTE: Consult the literature to ensure that random initialization is OK
        for L, N_w in enumerate(self.network_shape):
            self.biases[L] = np.random.randn(self.network_shape[L])

        return

    def __feed_forward(self, X_mb, y_mb):
        # Activation at the input layer
        z = X_mb @ self.weights[0] + self.biases[0]
        a = self.activation.evaluate(self.z)
        # Feed forward to compute z[L] a[L] for all layers
        for L in range(1, N_layers):
            z = a @ self.weights[L] + self.biases[L]
            a = self.activation.evaluate(self.z)

        # Compute the error of the output layer
        error_output = self.cost.evaluate_gradient(y, a)
        return

    def __feed_forward_output(self, X):
        # Feed-Forward to make predictions
        return

    def __backpropogation(self):
        return

    def train(self, M, learning_rate, n_epochs):
        # Ensure that the mini-batch size is NOT greater than
        assert M <= len(X)

        for epoch in range(n_epochs):
            # Pick out a new mini-batch
            mb = SGD.minibatch(X, m)
            for i in range(M):
                # with replacement, replace i with k
                # k = np.random.randint(M)
                self.__feed_forward(X[i], y[i])
                self.__backpropogation()
        return

    def predict(self, X):
        self.__feed_forward(X):
        return

    def __repr__(self):
        return f"FFNN: {self.N_layers} layers"


if __name__ == "__main__":
    # Define the network
    FFNN = FeedForwardNeuralNetwork(
        X = 1,
        y = 1,
        cost=CostFunctions.OLS,
        activation=ActivationFunctions.ReLU,
        activation_out=ActivationFunctions.Sigmoid,
        network_shape=[4, 5])

    #FFNN.train(X, y, M, learning_rate, n_epochs)
