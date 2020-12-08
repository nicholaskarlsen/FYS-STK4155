import numpy as np

import sys
sys.path.insert(0, "../../project_1/src")
sys.path.insert(0, "../../project_2/src")

from NeuralNetwork import FeedForwardNeuralNetwork
from SGD import minibatch

class DiffEqNeuralNetwork(FeedForwardNeuralNetwork):
	def __init__(self):

		"""
		Note: Some of the parameters used int he previous model are depricated, so perform a few
		clever (probably bug prone) work arounds that should not affect the current network
		while also keeping all the good stuff we had previously.
		"""
		super(DiffEqNeuralNetwork, self).__init__(
			X = X,
			Y = np.empty([0,0]), # Dont have/need output data -> pass a empty set to avoid errors.
			network_shape = network_shape,
			cost = cost,
			momentum = 0,
			lambd = 0,
			init_weights_method = None,
			learning_rate_decay = None,
		)

		return

    def __backpropogation(self, X_mb, M):
        for l in range(self.N_layers):
            self.error[l] = np.zeros([M, self.network_shape[l]])
        self.error[-1] = np.zeros([M, self.output_dim])

        # Compute the error at the output
        # TODO: SPECIALIZE THIS FOR ODE
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

	def train(self, N_minibatches, learning_rate, n_epochs):
		assert N_minibatches <= self.X.shape[0]
		# Increment the epoch counter
		self.total_epochs += n_epochs

		for epoch in range(n_epochs):
			# Pick out a new set of mini-batches
			mb = minibatch(self.X, N_minibatches)

			for i in range(N_minibatches):
                # with replacement, replace i with k
                # k = np.random.randint(M)
                X_mb = self.X[mb[i]]
                Y_mb = self.Y[mb[i]]
                M = X_mb.shape[0]  # Size of each minibach (NOT constant, see SGD.minibatch)
                # Feed-Forward to compute all the activations
                self.__feed_forward(X_mb, M)
                # Back-propogate to compute the gradients
                self.__backpropogation(X_mb, M)

		return


if __name__ == '__main__':
	NN = DiffEqNeuralNetwork()