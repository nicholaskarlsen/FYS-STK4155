import numpy as np
import matplotlib.pyplot as plt

import ActivationFunctions
import SGD
import CostFunctions
import sys


# Maps x -> scalar a
class Neuron:
    def __init__(self, activation_function, weights, bias):
        self.activation_function = activation_function
        self.w = weights
        self.bias = bias
        pass

    def __call__(self, x):
        # x = [1,x1,...]
        # w = [b,w1,...]
        z = np.dot(x.T, self.w)
        return self.activation_function(z)

    def __repr__(self):
        pass


class FeedForwardNeuralNetwork:
    def __init__(self, activation_function):

        if callable(activation_function):
            self.activation_function = activation_function
        else:
            raise Exception("activation_function is not a callable")

    def __initialize_weights(self):
        pass

    def add_layer(self, layer):
        self.layers.append(layer)
        return

    def train(self):
        pass

    def __backpropogation(self):


        pass

    def __call__(self):
        pass

    def __repr__(self):
        pass


if __name__ == "__main__":
    NN = FeedForwardNeuralNetwork(1)
