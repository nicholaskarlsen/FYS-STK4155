import sys
import numpy as np

sys.path.insert(0, "../src")
import NeuralNetwork as NN
import CostFunctions
import ActivationFunctions

"""
X = np.random.randn(100)
X = X.reshape(-1, 1)
Y = X * 2

n_epochs = 100

FFNN = NN.FeedForwardNeuralNetwork(
    X=X,
    Y=Y,
    cost=CostFunctions.SquareError,
    activation=ActivationFunctions.ID,
    activation_out=ActivationFunctions.ID,
    network_shape = [1,1,1]
    )
print("Check is products of weights => 2 for simple network shape:")
print("Network shape: [1,1,1]")
print("Product of initial weights = ", np.product(FFNN.weights))
FFNN.train(N_minibatches=32, learning_rate=0.0001, n_epochs=n_epochs)
print("Product of weights after {n_epochs} epochs = ", np.product(FFNN.weights))



print("\nCheck if network can doubble numbers for complex network shape:")
print("Network shape: [5, 2, 5, 1, 5, 2, 10]")
FFNN = NN.FeedForwardNeuralNetwork(
    X=X,
    Y=Y,
    cost=CostFunctions.SquareError,
    activation=ActivationFunctions.ID,
    activation_out=ActivationFunctions.ID,
    network_shape = [5, 2, 5, 1, 5, 2, 10]
    )



#print("Product of initial weights = ", np.product(FFNN.weights))
FFNN.train(N_minibatches=32, learning_rate=0.0001, n_epochs=n_epochs)
#print("Product of weights after {n_epochs} epochs = ", np.product(FFNN.weights))

inp = [[5],[10],[20],[25]]
print("input: ", inp)
print("predictions:", FFNN.predict(inp))
"""
#np.random.seed(2020)

X = np.random.uniform(2, 9, 1000)
X = X.reshape(-1, 1)
Y = X**2

FFNN = NN.FeedForwardNeuralNetwork(
    X=X,
    Y=Y,
    cost=CostFunctions.SquareError,
    activation=ActivationFunctions.LeakyReLU,
    activation_out=ActivationFunctions.LeakyReLU,
    network_shape = [10, 10]
    )

FFNN.train(N_minibatches=32, learning_rate=0.0001, n_epochs=1000)

inp = np.array([[2],[3],[4],[5],[6],[7],[8],[9]])
print("input: ", inp)
print("input: ", inp**2)
print("predictions:", FFNN.predict(inp))
