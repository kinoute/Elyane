from neuralnetwork.neural_network import NeuralNetwork
from layers.fc_layer import FCLayer
from layers.softmax_layer import SoftmaxLayer
from activations.tanh import TanH
from activations.relu import Relu
from activations.softmax import Softmax
from activations.sigmoid import Sigmoid
from losses.cross_entropy import CrossEntropy
from losses.multi_class_cross_entropy import MultiClassCrossEntropy
from losses.mse import MSE
import numpy as np
import pandas as pd

from keras.datasets import mnist
from keras.utils import np_utils

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
X = x_train.reshape(x_train.shape[0], 28*28).T
X = X.astype('float32')
X /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
Y = np_utils.to_categorical(y_train)


# Create our NN structure
net = NeuralNetwork()
net.add(FCLayer(28*28, 100, activation=TanH()))
net.add(FCLayer(100, 50, activation=TanH()))
net.add(FCLayer(50, 10, activation=Softmax()))

# train
net.use(loss=MultiClassCrossEntropy())
net.train(X[:, :2000], Y[0:2000].T, epochs=1000, learning_rate=1.2)

#print(net.get_acc(X, Y))

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

out = net.predict(x_test[0:10].T)
print("\n")
print("predicted values : ")
print(out, end="\n")
print(np.argmax(out, axis=0))
print("true values : ")
print(y_test[0:10])
print("true labels: ")
print(np.argmax(y_test[0:10],axis=1))

