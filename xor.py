from neuralnetwork.neural_network import NeuralNetwork
from layers.fc_layer import FCLayer
from activations.tanh import TanH
from activations.sigmoid import Sigmoid
from activations.softmax import Softmax
from losses.cross_entropy import CrossEntropy
import numpy as np

# Create our XOR dataset
X = np.random.randint(2, size = (1000, 2))

mask1 = X[:, 0] > 0.5
mask2 = X[:, 1] > 0.5

Y = np.logical_xor(mask1, mask2)
Y = Y.reshape(1000, 1)

X = X.T
Y = Y.T

# Create our NN structure
net = NeuralNetwork()
net.add(FCLayer(2, 5, activation=TanH()))
net.add(FCLayer(5, 2, activation=Sigmoid()))

# train
net.use(loss=CrossEntropy())
net.train(X, Y, epochs=1000, learning_rate=1.2)

# create our testing dataset
X_test = np.random.randint(2, size = (100, 2))

mask1 = X_test[:, 0] > 0.5
mask2 = X_test[:, 1] > 0.5

Y_test = np.logical_xor(mask1, mask2)
Y_test = Y_test.reshape(100, 1)

X_test = X_test.T
Y_test = Y_test.T

# testing!
result = net.predict(X_test)
result = np.where(result > 0.5, 1., 0.)[0]
print("Accuracy:", np.mean(result == Y_test), "out of 1")
