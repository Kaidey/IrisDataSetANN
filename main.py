import ANN as neuralNetwork
import numpy as np
import pandas as pd

np.random.seed(2)
network = neuralNetwork.Network()
iris = pd.read_csv('Iris.csv')


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

            res = 1 / float(1 + np.exp(- x))

            return res

        def derivative(x):
            return sigmoid(x) / float(1 - sigmoid(x))

        functions = {'function': np.vectorize(sigmoid),
                     'derivative': derivative}
    elif op == 2:

        def softmax(x):

            return np.exp(x) / float(sum(np.exp(x)))

        def derivative(x):

            nLayers = len(network.layers)
            outputLayer = network.layers[nLayers - 1]

            # Matrix with the same size as the weights matrix for the output layer
            adjustments = np.zeros((outputLayer.weights.shape[0], outputLayer.weights.shape[1]))

            dictKey = 'OutputLayer%i' % (nLayers - 1)
            outputPreviousLayer = network.results[dictKey]

            # for each output
            for i in range(len(x)):
                # for each input (connection) connected to ith output
                for j in range(outputPreviousLayer.shape[0]):
                    if i == j:
                        adjustments[i][j] = x[i] * (1 - outputPreviousLayer[j])
                    else:
                        adjustments[i][j] = - x[i] * outputPreviousLayer[j]

        functions = {'function': softmax,
                     'derivative': derivative}

    else:
        print('Invalid Option! \n')

    return functions


def displayCreateMenu():

    global network

    nLayers = input('Amount of layers (excluding input): \n')

    for i in range(nLayers):

        if i == (nLayers - 1):
            nNeurons = 3
            functions = selectFunctionMenu(i)
            previousLayer = network.layers[i - 1]
            nInputsPerNeuron = previousLayer.weights.shape[0]
            layer = neuralNetwork.NetworkLayer(nNeurons, nInputsPerNeuron, functions)
            network.addLayer(layer)

        else:
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

def prepareInput():

    iris.loc[iris['species'] == 'virginica', 'species'] = 2
    iris.loc[iris['species'] == 'versicolor', 'species'] = 1
    iris.loc[iris['species'] == 'setosa', 'species'] = 0

    trainingData = iris[['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth']].values
    labels = iris[['species']].values

    # Since we have 3 output neurons (3 classes)
    expectedOutput = np.zeros((len(trainingData), 3))

    # Create matrix of expected output
    # First column is 1 if expected output is setosa mapped to 0 above
    # Second column is 1 if expected output is versicolor mapped to 1 above
    # Third column is 1 if expected output is virginica mapped to 2 above
    # All others are 0
    for i in range(len(labels)):
        expectedOutput[i][labels[i]] = 1

    return trainingData, expectedOutput

def displayMainMenu():

    op = -1

    while op != 0:

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

        elif op == 2:

            if len(network.layers) == 0:
                print('Create an ANN first!')

            else:
                trainingData, expectedOutput = prepareInput()
                arr = np.array([5.1, 3.5, 1.4, 0.2])
                exp = np.array([1, 0, 0])
                network.trainANN(arr, exp, 2500)


                print(network.results)





displayMainMenu()
