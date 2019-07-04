from neuralnetwork.neural_network import NeuralNetwork

from layers.fc_layer import FCLayer
from layers.dropout import DropOut

from activations.tanh import TanH
from activations.sigmoid import Sigmoid

from optimizers.adam import Adam
from optimizers.rmsprop import RMSprop
from optimizers.momentum import Momentum
from optimizers.no_optim import NoOptim

from regularization.no_regularization import NoReg
from regularization.l2_regularization import L2Regularization

from losses.cross_entropy import CrossEntropy

from utils.create_xor_dataset import create_xor_dataset

import numpy as np

# get our XOR dataset
X, Y = create_xor_dataset(5000)

# Create our NN structure
net = NeuralNetwork()
net.add(FCLayer(2, 5, activation=TanH(), optimizer=Adam(), regularizer=NoReg()))
net.add(DropOut(keep_prob=1.0))
net.add(FCLayer(5, 2, activation=Sigmoid(), optimizer=Adam(), regularizer=NoReg()))

# train
net.use(loss=CrossEntropy())
net.train(X, Y, epochs=50, learning_rate=0.01, batch_size=256)

# training accuracy
train_results = net.predict(X)
train_results = np.where(train_results > 0.5, 1., 0.)[0]
print("Accuracy on training set:", np.mean(train_results == Y) * 100, "%")

# create our testing dataset
X_test, Y_test = create_xor_dataset(5000)

# testing accuracy
test_results = net.predict(X_test)
test_results = np.where(test_results > 0.5, 1., 0.)[0]
print("Accuracy on testing set:", np.mean(test_results == Y_test) * 100, "%")
