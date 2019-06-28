from neuralnetwork.neural_network import NeuralNetwork

from layers.fc_layer import FCLayer

from activations.tanh import TanH
from activations.sigmoid import Sigmoid

from losses.cross_entropy import CrossEntropy

from utils.create_xor_dataset import create_xor_dataset

import numpy as np

# get our XOR dataset
X, Y = create_xor_dataset(5000)

# Create our NN structure
net = NeuralNetwork()
net.add(FCLayer(2, 5, activation=TanH()))
net.add(FCLayer(5, 2, activation=Sigmoid()))

# train
net.use(loss=CrossEntropy())
train_results = net.train(X, Y, epochs=1000, learning_rate=1.2)

# accuracy
train_results = np.where(train_results > 0.5, 1., 0.)[0]
print("Accuracy on training set:", np.mean(train_results == Y) * 100, "%")

# create our testing dataset
X_test, Y_test = create_xor_dataset(5000)

# testing!
test_results = net.predict(X_test)
test_results = np.where(test_results > 0.5, 1., 0.)[0]
print("Accuracy on testing set:", np.mean(test_results == Y_test) * 100, "%")
