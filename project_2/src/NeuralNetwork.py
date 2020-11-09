import numpy as np
import matplotlib.pyplot as plt

# import SGD
import ActivationFunctions
import CostFunctions
import SGD
import sys

sys.path.insert(0, "../../project_1/src")


class FeedForwardNeuralNetwork:
    def __init__(self, X, Y, network_shape, activation, activation_out, cost, momentum=0, lambd=None):
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
        self.a = np.empty(self.N_layers + 1, dtype="object")  # Activation(z[l])
        self.z = np.empty(self.N_layers + 1, dtype="object")  # a[l-1] @ w[l]
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
        for i in range(1, self.N_layers):  # l = 1,..., L-1
            k = self.network_shape[i - 1]
            j = self.network_shape[i]
            self.weights[i] = np.random.randn(j, k)
        # Last hidden layer -> Output layer
        self.weights[-1] = np.random.randn(self.output_dim, self.network_shape[-1])

        return

    def __initialize_biases(self):
        for i in range(self.N_layers):
            # self.biases[i] = np.random.randn(self.network_shape[i])
            self.biases[i] = np.zeros(self.network_shape[i]) + 0.01
        # Last hidden layer -> Output layer
        # self.biases[-1] = np.random.randn(self.output_dim)
        self.biases[-1] = np.zeros(self.output_dim) + 0.01

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
        # self.error[-1] = self.cost.evaluate_gradient(
        #     self.a[-1], Y_mb
        # ) * self.activation_out.evaluate_derivative(self.z[-1])

        # Ad hoc way to deal with softmax: TODO!
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

    def __feed_forward_output(self, X_out):
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

    def train(self, N_minibatches, learning_rate, n_epochs):
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

                # TODO: ADD if for lambd != None to add penalty to gradients
                # Update the weights and biases using gradient descent
                for l in range(self.N_layers + 1):
                    # Change of weights
                    dw = self.weights[l] * self.momentum - learning_rate / M * self.cost_weight_gradient[l]
                    # Change of bias
                    db = self.biases[l] * self.momentum - learning_rate / M * self.cost_bias_gradient[l]
                    # Update weights and biases
                    self.weights[l] += dw
                    self.biases[l] += db
        return

    def predict(self, X_out):
        a_out = self.__feed_forward_output(X_out)
        return a_out

    def __repr__(self):
        return f"FFNN: {self.N_layers} layers"


class FFNNClassifier(FeedForwardNeuralNetwork):
    def __init__(self, X, Y, network_shape, activation, activation_out, cost, momentum=0, lambd=None):

        # Pre-proccess output data to onehot form
        Y_processed, labels = self.preprocess_classification_data(Y, return_labels=True)
        self.labels = labels
        # Let the constructor of the superclass handle everything else
        super().__init__(
            X, Y_processed, network_shape, activation, activation_out, cost, momentum=momentum, lambd=lambd
        )
        return

    def __backpropogation(self, X_mb, Y_mb, M):
        for l in range(self.N_layers):
            self.error[l] = np.zeros([M, self.network_shape[l]])
        self.error[-1] = np.zeros([M, self.output_dim])

        # Consider generalizing this!
        # Compute the error at the output
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

    def predict(self, X_out):
        a_out = super().predict(X_out)
        a_out = self.postprocess_classification_data(a_out, self.labels)
        return a_out

    def score(self, y_test, X_test):
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


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from FrankeFunction import *
    import linear_regression

    # NOTE: NN is meant to be magical black box that takes
    # (x, y) -> z. Why bother with an arbitrary design matrix itermediary step?

    # x = np.random.uniform(0, 1, 500)
    # y = np.random.uniform(0, 1, 500)
    N = 500
    xy = np.random.uniform(0, 1, [N, 2])
    z = FrankeFunction(xy[:, 0], xy[:, 0])
    z = z.reshape(-1, 1)

    xy_train, xy_test, z_train, z_test = train_test_split(xy, z, test_size=0.2)

    # deg = 6
    # X = linear_regression.design_matrix_2D(x, y, deg)

    # Define the network
    FFNN = FeedForwardNeuralNetwork(
        X=xy_train,
        Y=z_train,
        cost=CostFunctions.SquareError,
        activation=ActivationFunctions.Sigmoid,
        activation_out=ActivationFunctions.LeakyReLU,
        network_shape=[50, 50],
    )

    FFNN.train(N_minibatches=32, learning_rate=0.02, n_epochs=1000)
    z_test_prediction = FFNN.predict(xy_test)

    print(sum((z_test - z_test_prediction) ** 2) / N)

    # xv = np.random.uniform(0, 1, 50)
    # yv = np.random.uniform(0, 1, 50)
    # zv = FrankeFunction(x, y)
    # zv = z.reshape(-1,1)
    # X = linear_regression.design_matrix_2D(x, y, deg)

    """
    X = np.random.randn(1000)
    X = X.reshape(-1, 1)
    Y = X * 2


    FFNN = FeedForwardNeuralNetwork(X=X, Y=Y, cost=CostFunctions.SquareError,
        activation=ActivationFunctions.ID, activation_out=ActivationFunctions.ID,
        network_shape = [1]
        )

    FFNN.train(N_minibatches=32, learning_rate=0.001, n_epochs=10000)
    """
