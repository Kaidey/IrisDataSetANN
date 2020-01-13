
import numpy as np

class NetworkLayer:

    def __init__(self, nNeurons, nInputsPerNeuron, functions):

        self.weights = np.random.rand(nNeurons, nInputsPerNeuron)
        self.bias = np.ones(nNeurons)
        self.activation = functions['function']
        self.activationDerivative = functions['derivative']

    def activateLayer(self, x):

        return self.activation(x)

    def deriveActivation(self, x):

        return self.activationDerivative(x)


class Network:

    def __init__(self):

        self.layers = list()

    def addLayer(self, layer):

        self.layers.append(layer)
