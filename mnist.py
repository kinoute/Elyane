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

