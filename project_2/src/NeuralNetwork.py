import numpy as np
import matplotlib.pyplot as plt

#import SGD
import ActivationFunctions
import CostFunctions
import SGD
import sys
sys.path.insert(0, "../../project_1/src")


class FeedForwardNeuralNetwork:
    def __init__(self, X, Y, network_shape, activation, activation_out, cost, *lambd):
        """Models implements a Feed Forward Neural Network
        Args:
            network_shape (Array) : Defines the number of hidden layers (L-1) and number of neurons
                    (n) for each hidden layer in the network in the form [n_1, n_2, ..., n_L-1].

            activation (Object)   : The activation function; MUST be an implementation of the
                    ActivationFunction interface. See ActivationFunctions.py

            cost (Object)         : The cost function. Same as above; must be implementation of
                    CostFunction interface.
        """
        self.X = X
        self.Y = Y

        self.N_inputs, self.input_dim = self.X.shape
        self.N_outputs, self.output_dim = self.Y.shape

        # Make sure both data-sets are the same size
        assert self.N_inputs == self.N_outputs

        # Ensure that activation & cost are implementations of their respective interfaces
        # Which in turn guaranties the existence of the appropriate (static) methods
        if issubclass(activation, ActivationFunctions.ActivationFunction):
            self.activation = activation
        else:
            raise Exception("activation is not a sub-class of ActivationFunction")

        if issubclass(activation_out, ActivationFunctions.ActivationFunction):
            self.activation_out = activation
        else:
            raise Exception("activation_out is not a sub-class of ActivationFunction")

        if issubclass(cost, CostFunctions.CostFunction):
            self.cost = cost
        else:
            raise Exception("cost is not a sub-class of CostFunctions")

        # Initialize weight & bias
        self.network_shape = network_shape
        # Number of hidden layers in the network
        self.N_layers = len(self.network_shape)

        # Store weights and biases corresponding to L = 1, .., L-1, L
        # Where L + 1 goes from the last hidden layer to the output
        self.weights = np.empty(self.N_layers + 1, dtype="object")
        self.biases = np.empty(self.N_layers + 1, dtype="object")
        self.__initialize_weights()
        self.__initialize_biases()

        # Initialize array to store the gradients of the cost function wrt. weights & biases.
        self.cost_weight_gradient = np.empty(self.N_layers + 1, dtype="object")
        self.cost_bias_gradient = np.empty(self.N_layers + 1, dtype="object")
        # Initialize the shape based on weights & storages
        for i in range(self.N_layers):
            self.cost_weight_gradient[i] = np.empty(self.weights[i].shape)
            self.cost_bias_gradient[i] = np.empty(self.biases[i].shape)

        # Storage arrays for Feed-Forward & Backward propogation algorithm
        self.a = np.empty(self.N_layers + 1, dtype="object") # Activation(z[l])
        self.z = np.empty(self.N_layers + 1, dtype="object") # a[l-1] @ w[l]
        self.error = np.empty(self.N_layers + 1, dtype="object")

        # Note: These arrays will be re-used. But avoid re-initialization duiring
        # each itteration to reduce time spent on garbage collection
        return

    def __initialize_weights(self):
        # NOTE: Consult the literature to ensure that random initialization is OK
        # weight from k in l-1 to j in l -> w[l][j,k]
        # Input layer -> First Hidden layer
        k = self.input_dim
        j = self.network_shape[0]
        self.weights[0] = np.random.randn(j, k)
        # Hidden layers
        for i in range(1, self.N_layers): # l = 1,..., L-1
            k = self.network_shape[i-1]
            j = self.network_shape[i]
            self.weights[i] = np.random.randn(j, k)
        # Last hidden layer -> Output layer
        self.weights[-1] = np.random.randn(self.output_dim, self.network_shape[-1])

        return

    def __initialize_biases(self):
        for i in range(self.N_layers):
            self.biases[i] = np.random.randn(self.network_shape[i])
        # Last hidden layer -> Output layer
        self.biases[-1] = np.random.randn(self.output_dim)

        return

    def __feed_forward(self, X_mb, Y_mb, M):

        for l in range(self.N_layers):
            self.a[l] = np.zeros([M, self.network_shape[l]])
            self.z[l] = np.zeros([M, self.network_shape[l]])
        self.a[-1] = np.zeros([M, self.output_dim])
        self.z[-1] = np.zeros([M, self.output_dim])

        self.z[0] = X_mb @ self.weights[0].T + self.biases[0]
        self.a[0] = self.activation.evaluate(self.z[0])
        # Feed Forward
        # l = 2,3,4,...,L-1 compute z[l] = w[l] @ a[l-1] + b[l]
        for l in range(1, self.N_layers):
            self.z[l] = self.a[l-1] @ self.weights[l].T + self.biases[l]
            self.a[l] = self.activation.evaluate(self.z[l])
        # Note; weights = [1, ..., L, L+1] -> last index stores the output weight
        # Treat the output layer separately, due to different activation func

        self.z[-1] = self.a[-2] @ self.weights[-1].T + self.biases[-1]
        self.a[-1] = self.activation_out.evaluate(self.z[-1])
        return

    def __backpropogation(self, X_mb, Y_mb, M):
        for l in range(self.N_layers):
            self.error[l] = np.zeros([M, self.network_shape[l]])
        self.error[-1] = np.zeros([M, self.output_dim])

        # Compute the error at the output
        self.error[-1] = self.cost.evaluate_gradient(self.a[-1], Y_mb) \
                  * self.activation_out.evaluate_derivative(self.z[-1])

        # Backpropogate the error from l = L-1,...,2
        for l in range(self.N_layers-1, 0, -1):
            self.error[l] = (self.error[l+1] @ self.weights[l+1]) * self.activation.evaluate_derivative(self.z[l])


        self.cost_weight_gradient[0] = X_mb.T @ self.error[0]
        self.cost_bias_gradient[0] = np.sum(self.error[0], axis=1)

        print("-->", self.cost_weight_gradient[0].shape)
        print("-->", self.cost_weight_gradient[0].shape)

        # Compute the gradients
        for l in range(1, self.N_layers + 1):
            self.cost_weight_gradient[l] = self.a[l-1].T @ self.error[l]
            self.cost_bias_gradient[l] = np.sum(self.error[l], axis=1)

        return

    def __feed_forward_output(self, X):
        # Activation of the input layer
        z = self.weights[0] @ X + self.biases[0]
        a = self.activation.evaluate(z)

        # Activation of the hidden layers
        for L in range(1, self.N_layers - 1):
            z = self.weights[L] @ a + self.biases[L]
            a = self.activation.evaluate(z)

        z = self.weights[-1] @ a + self.biases[-1]
        a = self.activation_out(z)

        return a


    def train(self, N_minibatches, learning_rate, n_epochs):
        # Ensure that the mini-batch size is NOT greater than
        assert N_minibatches <= X.shape[0]

        for epoch in range(n_epochs):
            # Pick out a new mini-batch
            mb = SGD.minibatch(self.X, N_minibatches)
            for i in range(N_minibatches):
                # with replacement, replace i with k
                # k = np.random.randint(M)
                X_mb = self.X[mb[i]]
                Y_mb = self.Y[mb[i]]
                M = X_mb.shape[0] # Size of each minibach (NOT constant, see SGD.minibatch)


                # Feed-Forward to compute all the activations
                self.__feed_forward(X_mb, Y_mb, M)

                # Back-propogate to compute the gradients
                self.__backpropogation(X_mb, Y_mb, M)

                # TODO: ADD if for lambd != None to add penalty to gradients

                # Update the weights and biases using gradient descent
                for l in range(self.N_layers + 1):
                    print(self.weights[l].shape)
                    print(self.cost_weight_gradient[l].shape)


                    self.weights[l] -= learning_rate / M * self.cost_weight_gradient[l]
                    self.biases[l] -= learning_rate / M * self.cost_bias_gradient[l]
        return

    def predict(self, X):
        a = self.__feed_forward_output(X)
        return a

    def __repr__(self):
        return f"FFNN: {self.N_layers} layers"


if __name__ == "__main__":
    from FrankeFunction import *
    import linear_regression

    x = np.random.uniform(0, 1, 500)
    y = np.random.uniform(0, 1, 500)
    z = FrankeFunction(x, y)

    z = z.reshape(-1,1)

    deg = 2

    X = linear_regression.design_matrix_2D(x, y, deg)

    # Define the network
    FFNN = FeedForwardNeuralNetwork(
        X=X,
        Y=z,
        cost=CostFunctions.SquareError,
        activation=ActivationFunctions.ReLU,
        activation_out=ActivationFunctions.ReLU,
        network_shape=[4, 5, 6],
    )

    FFNN.train(N_minibatches=250, learning_rate=0.02, n_epochs=10)
