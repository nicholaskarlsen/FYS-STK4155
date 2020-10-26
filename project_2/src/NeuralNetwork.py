import numpy as np
import matplotlib.pyplot as plt


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
    def __init__(self):
        self.layers = []  # Layer of neurons
        pass

    def add_layer(self, layer):
        self.layers.append(layer)
        return

    def backpropogation(self):
        pass

    def __call__(self):
        pass

    def __repr__(self):
        pass
