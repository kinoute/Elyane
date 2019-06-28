from neuralnetwork.neural_network import NeuralNetwork
from layers.fc_layer import FCLayer
from layers.softmax_layer import SoftmaxLayer
from activations.tanh import TanH
from activations.relu import Relu
from activations.leaky_relu import LeakyRelu
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
net.add(FCLayer(50, 25, activation=TanH()))
net.add(SoftmaxLayer(25, 10, activation=Softmax()))

# train
net.use(loss=MultiClassCrossEntropy())
train_results = net.train(X[:,:20000], Y[0:20000].T, epochs=500, learning_rate=1.2)
print(train_results.shape)
print(Y[0:20000].T.shape)
train_results = np.argmax(train_results, axis = 1)
train_labels = np.argmax(Y[:,:20000].T, axis = 1)
# print(train_results.shape)
# print(train_labels.shape)
# print(train_results)
# print(train_labels)
# print("Accuracy on train set:", np.mean(train_results == train_labels), "out of 1")

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 28*28).T
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

out = net.predict(x_test)
print("\n")
print("predicted values : ")
print(out, end="\n")
print(np.argmax(out, axis=0))
print("true labels: ")
print(np.argmax(y_test,axis=1))
x = np.argmax(out, axis=0)
y = np.argmax(y_test,axis=1)
print("Accuracy on testing set:", np.mean(x == y), "out of 1")

