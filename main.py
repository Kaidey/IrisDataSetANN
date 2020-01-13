import ANN as neuralNetwork
import numpy as np

np.random.seed(2)
network = neuralNetwork.Network()


def selectFunctionMenu(layerID):
    functions = {}

    print('Choose the activation function for layer %i \n'
          '0 - Exit \n'
          '1 - Sigmoid \n'
          '2 - Softmax \n' % (layerID + 1))

    op = input('Choice: \n')

    if op == 0:
        exit(0)
    elif op == 1:

        def sigmoid(x):
            return 1 / float(1 + np.exp(- x))

        def derivative(x):

            return sigmoid(x) / float(1 - sigmoid(x))

        functions = {'function': sigmoid,
                     'derivative': derivative}
    elif op == 2:

        print('Not implemented yet! \n')

    else:
        print('Invalid Option! \n')

    return functions


def displayCreateMenu():

    global network

    nLayers = input('Amount of layers (excluding input): \n')

    for i in range(nLayers):

        nNeurons = input('Amount of neurons in layer %d: \n' % (i + 1))
        functions = selectFunctionMenu(i)

        # If it is the first layer, define no of inputs
        if i == 0:
            nInputsPerNeuron = input('Amount of input neurons: \n')
            layer = neuralNetwork.NetworkLayer(nNeurons, nInputsPerNeuron, functions)
            network.addLayer(layer)

        # Else, the program calculates no of inputs by itself
        else:
            previousLayer = network.layers[i - 1]
            nInputsPerNeuron = previousLayer.weights.shape[0]
            layer = neuralNetwork.NetworkLayer(nNeurons, nInputsPerNeuron, functions)
            network.addLayer(layer)

def displayMainMenu():
    print('0 - Exit \n'
          '1 - Create ANN \n'
          '2 - Train ANN \n'
          '3 - Make Predictions \n'
          '4 - Save Predictions to File \n'
          '5 - Statistics \n')

    op = input('Choice: \n')

    if op == 0:
        exit(0)

    elif op == 1:
        displayCreateMenu()


displayMainMenu()
