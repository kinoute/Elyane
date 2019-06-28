from neuralnetwork.neural_network import NeuralNetwork

from layers.fc_layer import FCLayer
from layers.softmax_layer import SoftmaxLayer

from activations.tanh import TanH
from activations.softmax import Softmax

from losses.multi_class_cross_entropy import MultiClassCrossEntropy

from utils.mnist_reader import load_mnist
from utils.one_hot_encoding import one_hot
from utils.normalize_images import normalize_images
import numpy as np

# load MNIST from gzip files
(x_train, y_train) = load_mnist('./datas/mnist/', 'train')
(x_test, y_test) = load_mnist('./datas/mnist/', 'test')

# reshape and normalize input data
x_train = normalize_images(x_train)

# number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = one_hot(y_train)

# number of classes / pixels per image
num_classes = y_train.shape[0]
num_pixels = x_train.shape[0]

# Create our NN structure
net = NeuralNetwork()
<<<<<<< HEAD
net.add(FCLayer(28*28, 100, activation=Relu()))
net.add(FCLayer(100, 50, activation=Relu()))
net.add(FCLayer(50, 25, activation=Relu()))
net.add(SoftmaxLayer(25, 10, activation=Softmax()))

# train
net.use(loss=MultiClassCrossEntropy())
train_results = net.train(X[:,:20000], Y[0:20000].T, epochs=1500, learning_rate=0.1)
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
=======
net.add(FCLayer(num_pixels, 100, activation=TanH()))
net.add(FCLayer(100, 50, activation=TanH()))
net.add(FCLayer(50, 25, activation=TanH()))
net.add(SoftmaxLayer(25, num_classes, activation=Softmax()))

# train
net.use(loss=MultiClassCrossEntropy())

train_results = net.train(x_train[:, :20000], y_train[:, :20000], epochs=1000, learning_rate=0.75)

# check training accuracy
train_results = np.argmax(train_results, axis = 0)
train_labels = np.argmax(y_train[:, :20000], axis = 0)
print("Accuracy on training set:", np.mean(train_results == train_labels) * 100, "%")

# Check our model on the test set
x_test = normalize_images(x_test)

test_results = net.predict(x_test)
test_results = np.argmax(test_results, axis = 0)

print("Accuracy on testing set:", np.mean(test_results == y_test) * 100, "%")
>>>>>>> 9e7b6c819aa039fc1cab161e62d32aef2c87c76d

