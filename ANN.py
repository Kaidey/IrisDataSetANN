
import numpy as np

class NetworkLayer:

    def __init__(self, nNeurons, nInputsPerNeuron, functions):

        self.weights = np.random.rand(nNeurons, nInputsPerNeuron)
        self.bias = np.random.rand(nNeurons)
        self.activation = np.vectorize(functions['function'])
        self.activationDerivative = np.vectorize(functions['derivative'])

    def activateLayer(self, x):

        return self.activation(x)

    def deriveActivation(self, x):

        return self.activationDerivative(x)


class Network:

    def __init__(self):

        self.layers = list()
        self.results = {}

    def addLayer(self, layer):

        self.layers.append(layer)

    # Input data is an numberOfCases x nOfAttributes matrix
    # Weights is a nNeurons x nInputs matrix (nInputs for layer 1 is the nOfAttributes)
    # Bias is a 1 x nNeurons matrix (1 bias for each neuron)
    # WeightedSums is the result of the dot product of input with weights + bias resulting in a numberOfCases x nNeurons matrix
    # LayerOutput is a matrix the same size as WeightedSums, having all WeightedSums now passed through the activation function
    def forwardPropagation(self, inputData):

        results = {}

        for i in range(len(self.layers)):
            layer = self.layers[i]

            # Input for layer 1 is the network's training data
            if i == 0:
                # Originally weights is a nNeurons x nOfAttributes so to be able to do the dot product we need to transpose it -> numberOfCases x nOfAttributes . nOfAttributes x nNeurons
                weightedSums = np.dot(inputData, layer.weights.T)
                output = layer.activateLayer(weightedSums)

            else:
                # Input for next layers is the output from the previous layer
                inputDictKey = 'OutputLayer%d' % i
                inputData = results[inputDictKey]
                weightedSums = np.dot(inputData, layer.weights.T)
                output = layer.activateLayer(weightedSums)

            dictionaryKeyWS = 'WeightedSumsLayer%d' % (i + 1)
            dictionaryKeyOut = 'OutputLayer%d' % (i + 1)

            results[dictionaryKeyWS] = weightedSums
            results[dictionaryKeyOut] = output

        self.results = results
