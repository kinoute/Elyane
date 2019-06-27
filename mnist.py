from neuralnetwork.neural_network import NeuralNetwork
from layers.fc_layer import FCLayer
from layers.softmax_layer import SoftmaxLayer
from activations.tanh import TanH
from activations.relu import Relu
from activations.softmax import Softmax
from losses.cross_entropy import CrossEntropy
from losses.multi_class_cross_entropy import MultiClassCrossEntropy
from losses.mse import MSE
import numpy as np
import pandas as pd


def one_hot_encoding(y, n_examples, n_classes):
    """ One hot Encode les labels y.
    Arguments:
        y : dataset de label
        n_examples : nombre d'exemples dans y
        n_classes  : nombre de classes
    """
    one_hot = np.eye(n_classes)
    Y_new = one_hot[y.astype('int32')]
    return Y_new.T.reshape(n_classes, n_examples)

mnist = pd.read_csv('datas/mnist/mnist_train.csv')[:1500]

print("shape at first", mnist.shape)

train_label = mnist["label"]
mnist.drop(['label'], inplace = True, axis = 1)
mnist /= 255
mnist = mnist.values.reshape(784, 1500)

print("shape after drop", mnist.shape)

print("shape after, reshape -1", mnist.shape)

train_label = train_label.values.flatten()
classes = np.unique(train_label)

X = mnist
Y = one_hot_encoding(train_label, mnist.shape[1], len(classes))

print("X shape after T", X.shape)
print("Y shape after T", Y.shape)


# Create our NN structure
net = NeuralNetwork()
net.add(FCLayer(X.shape[0], 100, activation=TanH()))
net.add(FCLayer(100, 50, activation=TanH()))
net.add(FCLayer(50, 20, activation=TanH()))
net.add(SoftmaxLayer(20, Y.shape[0], activation=Softmax()))

# train
net.use(loss=MultiClassCrossEntropy())
net.train(X, Y, epochs=1000, learning_rate=1.2)

# test

mnist_test = pd.read_csv('datas/mnist/mnist_test.csv')[:1500]

test_label = mnist_test["label"]
mnist_test.drop(['label'], inplace = True, axis = 1)
mnist_test /= 255
mnist_test = mnist_test.values.reshape(784, 1500)

print("shape after drop", mnist_test.shape)

print("shape after, reshape -1", mnist_test.shape)

test_label = test_label.values.flatten()
test_classes = np.unique(test_label)

X_test = mnist_test
Y_test = one_hot_encoding(test_label, mnist_test.shape[1], len(test_classes))

print("X_test shape before prediction", X_test.shape)
print("Y_test shape before prediction", Y_test.shape)

result = net.predict(X_test[:, :3])

print("ok", result)
print(np.argmax(result, axis = 0))
print(Y_test[:, :3])
print(test_label[: , :3])
