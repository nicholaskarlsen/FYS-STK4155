import numpy as np
import matplotlib.pyplot as plt

import ActivationFunctions
import CostFunctions
import SGD
import sys

sys.path.insert(0, "../../project_1/src")

from stat_tools import MSE, R2


class FeedForwardNeuralNetwork:
    def __init__(
        self,
        X,
        Y,
        network_shape,
        activation,
        activation_out,
        cost=CostFunctions.SquareError,
        momentum=0,
        lambd=None,
        init_weights_method=None,
        learning_rate_decay=None,
    ):
        """Implements a Feed Forward Neural Network

        NOTE: Elaborate as to how this class can also handle
        the binary classification problem. Use sigmoid as
        activation & cross entropy as cost.

        Args:
            X (Array) : Input data to train the network on. Data is expected to be structured in a
                Row-Major fashion; that is, the expected shape is [N data points, Dimensionality of data]

            Y (Array) : Output data corrsponding to the X data set and structured in a similar fashion

            network_shape (Array) : Defines the number of hidden layers (L-1) and number of neurons (n)
                for each hidden layer in the network in the form [n_1, n_2, ..., n_L-1].

            activation (Object) : The activation function; MUST be an implementation of the
                ActivationFunction interface. See ActivationFunctions.py

            cost (Object) : The cost function. Same as above; must be implementation of
                CostFunction interface.

            momentum (Float): Hyperparameter determining the momentum to be used in the Stochastic
                gradient descent.

            lambd (Float): Hyper parameter corresponding to a L2 penalty in the cost function akin
                to that of Ridge regression.

            init_weights_method (String): Parameter determining the type of initialization to be used for
                the weights. If none, or an invalid value is choosen it will default to sampling
                weights from the standard normal distribution N(0,1).
                Available options are: ["xavier", "he"]

            learning_rate_decay (String): Choose which method to decay the learning rate.
                'inverse' or 'exponential'. If none are choosen; keep it constant.
        """
        self.X = X
        self.Y = Y

        self.N_inputs, self.input_dim = self.X.shape
        self.N_outputs, self.output_dim = self.Y.shape

        # Make sure both data-sets are the same size
        assert self.N_inputs == self.N_outputs

        self.lambd = lambd
        self.momentum = momentum

        # Keep track of total number of epochs the network has been trained for
        self.total_epochs = 0

        # Ensure that activation & cost are implementations of their respective interfaces
        # Which in turn guaranties the existence of the appropriate (static) methods
        if issubclass(activation, ActivationFunctions.ActivationFunction):
            self.activation = activation
        else:
            raise Exception("activation is not a sub-class of ActivationFunction")

        if issubclass(activation_out, ActivationFunctions.ActivationFunction):
            self.activation_out = activation_out
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

        # Initialize the weights based on the users input. If no preference; sample from N(0,1)
        if init_weights_method == "he":
            self.__he_initialize_weights()
        elif init_weights_method == "xavier":
            self._xavier_initialize_weights()
        else:
            self.__initialize_weights()

        # Initialize the bias as zero
        self.__initialize_biases()

        # Initialize array to store the gradients of the cost function wrt. weights & biases.
        self.cost_weight_gradient = np.empty(self.N_layers + 1, dtype="object")
        self.cost_bias_gradient = np.empty(self.N_layers + 1, dtype="object")
        # Initialize the shape based on weights & storages
        for i in range(self.N_layers):
            self.cost_weight_gradient[i] = np.empty(self.weights[i].shape)
            self.cost_bias_gradient[i] = np.empty(self.biases[i].shape)

        # Storage arrays for Feed-Forward & Backward propogation algorithm
        self.a = np.empty(self.N_layers + 1, dtype="object")  # Activation(z[l])
        self.z = np.empty(self.N_layers + 1, dtype="object")  # a[l-1] @ w[l]
        self.error = np.empty(self.N_layers + 1, dtype="object")

        # Note: These arrays will be re-used. But avoid re-initialization duiring
        # each itteration to reduce time spent on garbage collection

        # Assign learning decay method
        if learning_rate_decay == "exponential":
            self.learning_rate_func = self.__exponential_decay
        elif learning_rate_decay == "inverse":
            self.learning_rate_func = self.__inverse_decay
        else:
            self.learning_rate_func = self.__constant_rate
        return

    def __initialize_weights(self):
        # Standard, initialization; sample from a normal distribution.
        # weight from k in l-1 to j in l -> w[l][j,k]
        # Input layer -> First Hidden layer
        print("Initializing weights using: Normal distribution")
        k = self.input_dim
        j = self.network_shape[0]
        self.weights[0] = np.random.randn(j, k)
        # Hidden layers
        for i in range(1, self.N_layers):  # l = 1,..., L-1
            k = self.network_shape[i - 1]
            j = self.network_shape[i]
            self.weights[i] = np.random.randn(j, k)
        # Last hidden layer -> Output layer
        self.weights[-1] = np.random.randn(self.output_dim, self.network_shape[-1])

        return

    def __he_initialize_weights(self):
        # He initialization; suitable to pair with ReLU
        # weight from k in l-1 to j in l -> w[l][j,k]
        # Input layer -> First Hidden layer
        print("Initializing weights using: He")
        k = self.input_dim
        j = self.network_shape[0]
        self.weights[0] = np.random.randn(j, k) * np.sqrt(2) / np.sqrt(k)
        # Hidden layers
        for i in range(1, self.N_layers):  # l = 1,..., L-1
            k = self.network_shape[i - 1]
            j = self.network_shape[i]
            self.weights[i] = np.random.randn(j, k) * np.sqrt(2) / np.sqrt(k)
        # Last hidden layer -> Output layer
        k = self.network_shape[-1]
        j = self.output_dim
        self.weights[-1] = np.random.randn(j, k) * np.sqrt(2) / np.sqrt(k)

        return

    def _xavier_initialize_weights(self):
        # Xavier initialization; Sample from a uniform distribution
        print("Initializing weights using: Xavier")
        # Define a lambda to easily compute the lower/upper bounds of the distribution
        sample_bound = lambda j, k: np.sqrt(6) / np.sqrt(j + k)
        k = self.input_dim
        j = self.network_shape[0]
        self.weights[0] = np.random.uniform(-sample_bound(j, k), sample_bound(j, k), size=(j, k))
        # Hidden layers
        for i in range(1, self.N_layers):  # l = 1,..., L-1
            k = self.network_shape[i - 1]
            j = self.network_shape[i]
            self.weights[i] = np.random.uniform(-sample_bound(j, k), sample_bound(j, k), size=(j, k))
        # Last hidden layer -> Output layer
        k = self.network_shape[-1]
        j = self.output_dim
        self.weights[-1] = np.random.uniform(-sample_bound(j, k), sample_bound(j, k), size=(j, k))

        return

    def __initialize_biases(self):
        for i in range(self.N_layers):
            # self.biases[i] = np.random.randn(self.network_shape[i])
            self.biases[i] = np.zeros(self.network_shape[i])  # + 0.01
        # Last hidden layer -> Output layer
        # self.biases[-1] = np.random.randn(self.output_dim)
        self.biases[-1] = np.zeros(self.output_dim)  # + 0.01

        return

    def __feed_forward(self, X_mb, M):

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
            self.z[l] = self.a[l - 1] @ self.weights[l].T + self.biases[l]
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
        # Normal way to deal with everything else. TODO!
        self.error[-1] = self.cost.evaluate_gradient(
            self.a[-1], Y_mb
        ) * self.activation_out.evaluate_derivative(self.z[-1])

        # Ad hoc way to deal with softmax: TODO!
        # self.error[-1] = self.a[-1] - Y_mb

        # Backpropogate the error from l = L-1,...,1
        for l in range(self.N_layers - 1, -1, -1):
            self.error[l] = (self.error[l + 1] @ self.weights[l + 1]) * self.activation.evaluate_derivative(
                self.z[l]
            )

        self.cost_weight_gradient[0] = self.error[0].T @ X_mb
        self.cost_bias_gradient[0] = np.sum(self.error[0], axis=0)

        # Compute the gradients
        for l in range(1, self.N_layers + 1):
            self.cost_weight_gradient[l] = self.error[l].T @ self.a[l - 1]
            self.cost_bias_gradient[l] = np.sum(self.error[l], axis=0)

        if self.lambd != None:
            for l in range(self.N_layers + 1):
                self.cost_weight_gradient[l] += self.lambd * self.weights[l]

        return

    def __feed_forward_output(self, X_out):
        """
        Don't need to store all of the activations when making predictions, so created a "duplicate"
        of feed forward method that is used specifically for this purpose
        """
        z_out = X_out @ self.weights[0].T + self.biases[0]
        a_out = self.activation.evaluate(z_out)
        # Feed Forward
        # l = 2,3,4,...,L-1 compute z[l] = w[l] @ a[l-1] + b[l]
        for l in range(1, self.N_layers):
            z_out = a_out @ self.weights[l].T + self.biases[l]
            a_out = self.activation.evaluate(z_out)
        # Note; weights = [1, ..., L, L+1] -> last index stores the output weight
        # Treat the output layer separately, due to different activation func
        z_out = a_out @ self.weights[-1].T + self.biases[-1]
        a_out = self.activation_out.evaluate(z_out)

        return a_out

    def __constant_rate(self, init_learning_rate, decay_rate):
        return init_learning_rate

    def __exponential_decay(self, init_learning_rate, decay_rate):
        return init_learning_rate * np.exp(-decay_rate * self.total_epochs)

    def __inverse_decay(self, init_learning_rate, decay_rate):
        return init_learning_rate / (1 + decay_rate * self.total_epochs)

    def train(self, N_minibatches, learning_rate, n_epochs, decay_rate=0):
        # Ensure that the mini-batch size is NOT greater than
        assert N_minibatches <= self.X.shape[0]
        # Increment the epoch counter
        self.total_epochs += n_epochs

        for epoch in range(n_epochs):
            # Pick out a new mini-batch
            mb = SGD.minibatch(self.X, N_minibatches)
            for i in range(N_minibatches):
                # with replacement, replace i with k
                # k = np.random.randint(M)
                X_mb = self.X[mb[i]]
                Y_mb = self.Y[mb[i]]
                M = X_mb.shape[0]  # Size of each minibach (NOT constant, see SGD.minibatch)
                # Feed-Forward to compute all the activations
                self.__feed_forward(X_mb, M)
                # Back-propogate to compute the gradients
                self.__backpropogation(X_mb, Y_mb, M)

                # update the learning rate using the choosen rate using the chosen method
                lr = self.learning_rate_func(learning_rate, decay_rate)
                # Update the weights and biases using gradient descent
                for l in range(self.N_layers + 1):
                    # Change of weights
                    dw = self.weights[l] * self.momentum - lr / M * self.cost_weight_gradient[l]
                    # Change of bias
                    db = self.biases[l] * self.momentum - lr / M * self.cost_bias_gradient[l]
                    # Update weights and biases
                    self.weights[l] += dw
                    self.biases[l] += db
        return

    def predict(self, X_out):
        a_out = self.__feed_forward_output(X_out)
        return a_out

    def score(self, y_test, X_test, metric="default"):
        # Compute the MSE (or R2) of the model wrt. testing data X_test (input), y_test (output)
        y_test_model = self.predict(X_test)

        if metric == "r2":
            return R2(y_test, y_test_model)
        else:
            return MSE(y_test, y_test_model)

    def __repr__(self):
        return f"FFNN: {self.N_layers} layers"


class FFNNClassifier(FeedForwardNeuralNetwork):
    def __init__(
        self,
        X,
        Y,
        network_shape,
        activation,
        activation_out=ActivationFunctions.Softmax,
        momentum=0,
        lambd=None,
        init_weights_method=None,
        learning_rate_decay=None,
    ):
        """
        Specialized FFNN that performs classification using either SoftMax or sigmoid activations,
        where the backpropogation method has been overwritten to do this efficiently (see report).

        Note: Because of some rather odd behaviour of python when it comes to utilizing overwritten
            methods properly; we copy-pasted the train and feedforward methods into this class,
            as a quick and dirty fix. These two methods are thus unchanged.

        For documentation of all the parameters; please refer to the superclass.
        """

        # Pre-proccess output data to onehot form
        Y_processed, labels = self.preprocess_classification_data(Y, return_labels=True)
        self.labels = labels

        # Let the constructor of the superclass handle everything else
        super().__init__(
            X,
            Y_processed,
            network_shape,
            activation,
            activation_out,
            cost=CostFunctions.CostFunction,  # Send an empty iterface; since this should NOT be called.
            momentum=momentum,
            lambd=lambd,
            init_weights_method=init_weights_method,
            learning_rate_decay=learning_rate_decay,
        )
        return

    def __feed_forward(self, X_mb, M):

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
            self.z[l] = self.a[l - 1] @ self.weights[l].T + self.biases[l]
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

        # Compute the error at the output (This line is specialized in the override! see report)
        self.error[-1] = self.a[-1] - Y_mb

        # Backpropogate the error from l = L-1,...,1
        for l in range(self.N_layers - 1, -1, -1):
            self.error[l] = (self.error[l + 1] @ self.weights[l + 1]) * self.activation.evaluate_derivative(
                self.z[l]
            )

        self.cost_weight_gradient[0] = self.error[0].T @ X_mb
        self.cost_bias_gradient[0] = np.sum(self.error[0], axis=0)

        # Compute the gradients
        for l in range(1, self.N_layers + 1):
            self.cost_weight_gradient[l] = self.error[l].T @ self.a[l - 1]
            self.cost_bias_gradient[l] = np.sum(self.error[l], axis=0)

        if self.lambd != None:
            for l in range(self.N_layers + 1):
                self.cost_weight_gradient[l] += self.lambd * self.weights[l]

        return

    def train(self, N_minibatches, learning_rate, n_epochs, decay_rate=0):
        # Ensure that the mini-batch size is NOT greater than
        assert N_minibatches <= self.X.shape[0]
        # Increment the epoch counter
        self.total_epochs += n_epochs

        for epoch in range(n_epochs):
            # Pick out a new mini-batch
            mb = SGD.minibatch(self.X, N_minibatches)
            for i in range(N_minibatches):
                # with replacement, replace i with k
                # k = np.random.randint(M)
                X_mb = self.X[mb[i]]
                Y_mb = self.Y[mb[i]]
                M = X_mb.shape[0]  # Size of each minibach (NOT constant, see SGD.minibatch)
                # Feed-Forward to compute all the activations
                self.__feed_forward(X_mb, M)
                # Back-propogate to compute the gradients
                self.__backpropogation(X_mb, Y_mb, M)

                # update the learning rate using the choosen rate using the chosen method
                lr = self.learning_rate_func(learning_rate, decay_rate)
                # Update the weights and biases using gradient descent
                for l in range(self.N_layers + 1):
                    # Change of weights
                    dw = self.weights[l] * self.momentum - lr / M * self.cost_weight_gradient[l]
                    # Change of bias
                    db = self.biases[l] * self.momentum - lr / M * self.cost_bias_gradient[l]
                    # Update weights and biases
                    self.weights[l] += dw
                    self.biases[l] += db
        return

    def predict(self, X_out, return_probabilities=False):
        a_out = super().predict(X_out)
        a_labels = self.postprocess_classification_data(a_out, self.labels)

        if return_probabilities:
            return a_labels, a_out

        return a_labels

    def score(self, y_test, X_test):
        # Compute the accuracy score of the model wrt. testing data X_test (input), y_test (output)
        y_test_model = self.predict(X_test)
        return sum(np.equal(y_test, y_test_model)) / len(y_test)

    @staticmethod
    def preprocess_classification_data(Y, labels=None, return_labels=False):
        """Preprocesses a dataset for classification. transforming a one dimensional set of labels
        into matrix of shape = [Number of labels, Number of different labels] i.e
                        [[1, 0, 0],
        [1, 2, 3, 1] ->  [0, 1, 0],
                         [0, 0, 1],
                         [1, 0, 0]]
        with automatic detection of labels; assuming all the different types of labels are contained
        within the data set. Alternatively; an array of labels i.e [1,2,3] may be provided for
        both speed and convenience.

        args:
            Y (Array) - Discrete set of labels
            labels (Array) - List of labels present in Y.
        """

        # Generate labels if not supplied
        if labels is None:
            labels = []
            for element in Y:
                if element not in labels:
                    labels.append(element)
            # Covert to numpy array & sort
            labels = np.sort(np.array(labels))

        # Separate Y to binary response arrays where the second index
        # Corresponds to the index of the labels list
        Y_processed = np.zeros([len(Y), len(labels)])
        for i, element in enumerate(Y):
            Y_processed[i, np.where(labels == element)] = 1

        if return_labels:
            return Y_processed, labels

        return Y_processed

    @staticmethod
    def postprocess_classification_data(Y, labels):
        """Transforms processed data in the form generated by preprocess_classification_data
        back to labeled form.
        args:
            Y (array) - Classification data in onehot form
            labels (array) - labels corresponding to the onehot form of Y
        """
        return labels[np.argmax(Y, axis=1)]
