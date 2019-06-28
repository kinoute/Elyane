import numpy as np
import math
class NeuralNetwork:

    def __init__(self):
        self.layers = []
        self.loss = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # add loss function to network
    def use(self, loss):
        self.loss = loss

    def cost(self, loss, size):
        return (np.sum(loss) / size)

    def train(self, X, Y, epochs, learning_rate, batch_size = 128):

        # Number of samples in our training set
        train_size = X.shape[1]
        batch_size = batch_size if batch_size else train_size

        for i in range(epochs):

            # At first, A0 equals to the training set
            A = X

            # Forward Propagation
            for layer in self.layers:
                A = layer.forward_pass(A)

            print("cost after", i, "iterations:", self.cost(self.loss.fct(Y, A), train_size))

            deriv_activation = self.loss.deriv(Y, A)

            # Backward propagation
            for layer in reversed(self.layers):
                deriv_activation = layer.backward_pass(deriv_activation, learning_rate, train_size)

        return A

    def predict(self, X):

        # first activation is our training set
        A = X

        # Number of classes to predict
        #
        for layer in self.layers:
            A = layer.forward_pass(A)

        return A
