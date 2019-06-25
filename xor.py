from neuralnetwork.neural_network import NeuralNetwork
from layers.fc_layer import FCLayer
from activations.tanh import tanH
from activations.sigmoid import Sigmoid
from losses.cross_entropy import CrossEntropy
import numpy as np

X = np.random.randint(2, size = (100, 2))

mask1 = X[:, 0] > 0.5
mask2 = X[:, 1] > 0.5

Y = np.logical_xor(mask1, mask2)

Y = Y.reshape(100, 1)

X = X.T
Y = Y.T

# network
net = NeuralNetwork()
net.add(FCLayer(2, 3, tanH()))
net.add(FCLayer(3, 1, Sigmoid()))

# train
net.use(CrossEntropy)
net.train(X, Y, epochs=1000, learning_rate=1.2)

# test
#out = net.predict(x_train)
#print(out)
