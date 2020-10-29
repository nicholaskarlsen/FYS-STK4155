import numpy as np
import matplotlib.pyplot as plt

import SGD
import ActivationFunctions
import CostFunctions
import sys


class FeedForwardNeuralNetwork:
    def __init__(self, X, Y, network_shape, activation, activation_out, cost, *lambd):
        """Models implements a Feed Forward Neural Network
        Args:
            network_shape (Array) : Defines the number of hidden layers (L) and number of neurons
                    (n) for each hidden layer in the network in the form [n_1, n_2, ..., n_L].

            activation (Object)   : The activation function; MUST be an implementation of the
                    ActivationFunction interface. See ActivationFunctions.py

            cost (Object)         : The cost function. Same as above; must be implementation of
                    CostFunction interface.
        """
        self.X = X
        self.Y = Y

        try:
            self.N_inputs, self.input_dim = self.X.shape
        except ValueError as e:  # if 1D input data
            self.N_inputs = len(self.X)
            self.input_dim = 1

        try:
            self.N_outputs, self.output_dim = self.Y.shape
        except ValueError as e:  # if 1D output data
            self.N_outputs = len(self.Y)
            self.output_dim = 1

        # Make sure both data-sets are the same size
        assert self.N_inputs == self.N_outputs

        # Ensure that activation & cost are implementations of their respective interfaces
        # Which in turn guaranties the existence of the appropriate (static) methods
        if issubclass(activation, ActivationFunctions.ActivationFunction):
            self.activation = activation
        else:
            raise Exception("activation is not a sub-class of ActivationFunction")

        if issubclass(activation_out, ActivationFunction):
            self.activation_out = activation
        else:
            raise Exception("activation_out is not a sub-class of ActivationFunction")

        if issubclass(cost, CostFunctions.CostFunction):
            self.cost = cost
        else:
            raise Exception("cost is not a sub-class of CostFunctions")

        # Initialize weight & bias
        self.network_shape = network_shape
        self.N_hidden_layers = len(self.network_shape)     # Number of hidden layers in the network
        self.N_layers = self.N_hidden_layers
        self.weights = np.empty(self.N_layers, dtype="object")
        self.biases = np.empty(self.N_layers, dtype="object")
        self.__initialize_weights()
        self.__initialize_biases()

        # Storage arrays for Feed-Forward & Backward propogation algorithm
        self.a = np.empty(self.N_layers, dtype="object")
        self.z = np.empty(self.N_layers, dtype="object")

        self.lambd = lambd

        return

    def __initialize_weights(self):
        # NOTE: Consult the literature to ensure that random initialization is OK
        # TODO: Look more closely at this
        # weight from k in l-1 to j in l -> w[l][j,k]
        # Input layer -> First Hidden layer
        self.weights[0] = np.random.randn([self.N_inputs, self.network_shape[0]])
        # Hidden layers
        for L in range(1, self.N_layers - 1):
            self.weights[L] = np.random.randn([j, k])
        # Last hidden layer -> Output layer
        self.weights[-1] = np.randn([self.network_shape[-1], self.N_outputs])

        return

    def __initialize_biases(self):
        # NOTE: Consult the literature to ensure that random initialization is OK
        # TODO: Look more closely at this, only a sketch for now.
        self.biases[0] = np.random.randn(self.network_shape[0])
        # Hidden layers
        for L in range(1, self.N_layers - 1):
            self.biases[L] = np.random.randn(self.network_shape[L])
        # Last hidden layer -> Output layer
        self.biases[-1] = np.randn(self.output_dim)

        return

    def __feed_forward(self, X_mb, Y_mb):
        # Activation at the input layer
        self.z[0] = X_mb @ self.weights[0] + self.biases[0]
        self.a[0] = self.activation.evaluate(self.z[0])

        # Feed Forward
        # l = 2, 3, ..., L compute z[l] = w[l] @ a[l-1] + b[l]
        for l in range(1, self.N_layers - 1):
            self.z[l] = self.a[l-1] @ self.weights[l] + self.biases[l]
            self.a[l] = self.activation.evaluate(z[l])

        # Treat the output layer separately, due to different activation func
        self.z[-1] = self.a[-2] @ self.weights[-1] + self.biases[-1]
        self.a[-1] = self.activation_out.evaluate(self.z[-1])
        return

    def __backpropogation(self, X_mb, Y_mb):
        error = np.empty(self.N_hidden_layers, dtype="Object")

        # Compute the error at the output
        error[-1] = self.cost.evaluate_gradient(Y_mb, self.a[-1]) \
                  * self.activation_out.evaluate_derivative(self.z[-1])

        # Backpropogate the error from L-1, ..., 2
        for l in range(self.N_hidden_layers-2, 1, -1):
            error[l] = (self.w[l+1].T @ error[l+1]) * self.activation.evaluate_derivative(self.z[l])

        # Compute gradients as
        dCdw[l][j][k] = a[l-1][k] @ error[l][j]
        dCdb[l][j] = error[l][j]

        return

    def __feed_forward_output(self, X):
        # Activation of the input layer
        z = X @ self.weights[0] + self.biases[0]
        a = self.activation.evaluate(z)

        # Activation of the hidden layers
        for L in range(1, self.N_layers - 1):
            z = a @ self.weights[L] + self.biases[L]
            a = self.activation.evaluate(z)

        z = a @ self.weights[-1] + self.biases[-1]
        a = self.activation_out(z)

        return a


    def train(self, M, learning_rate, n_epochs):
        # Ensure that the mini-batch size is NOT greater than
        assert M <= len(X)

        for epoch in range(n_epochs):
            # Pick out a new mini-batch
            mb = SGD.minibatch(X, m)
            for i in range(M):
                # with replacement, replace i with k
                # k = np.random.randint(M)
                self.__feed_forward(self.X[i], self.Y[i])
                self.__backpropogation(self.X[i], self.Y[i])
        return

    def predict(self, X):
        a = self.__feed_forward_output(X):
        return a

    def __repr__(self):
        return f"FFNN: {self.N_layers} layers"


if __name__ == "__main__":
    # Define the network
    FFNN = FeedForwardNeuralNetwork(
        X=1,
        y=1,
        cost=CostFunctions.OLS,
        activation=ActivationFunctions.ReLU,
        activation_out=ActivationFunctions.Sigmoid,
        network_shape=[4, 5],
    )

    # FFNN.train(X, y, M, learning_rate, n_epochs)
