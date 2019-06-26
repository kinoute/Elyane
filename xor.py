from neuralnetwork.neural_network import NeuralNetwork
from layers.fc_layer import FCLayer
from activations.tanh import TanH
from activations.sigmoid import Sigmoid
from activations.softmax import Softmax
from losses.cross_entropy import CrossEntropy
from losses.mse import MSE
from losses.mae import MAE
import numpy as np

# Create our XOR dataset
X = np.random.randint(2, size = (100, 2))

mask1 = X[:, 0] > 0.5
mask2 = X[:, 1] > 0.5

Y = np.logical_xor(mask1, mask2)
Y = Y.reshape(100, 1)

X = X.T
Y = Y.T

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


print(Y.shape)
print(Y)
print("-----------------")
Y = one_hot_encoding(Y, 100, 2)
print(Y)
print(Y.shape)

# Create our NN structure
net = NeuralNetwork()
net.add(FCLayer(2, 3, activation=TanH()))
net.add(FCLayer(3, 2, activation=Softmax()))

# train
net.use(loss=CrossEntropy())
net.train(X, Y, epochs=1000, learning_rate=1.2)

# test
result = net.predict(X, Y)
print(result)
