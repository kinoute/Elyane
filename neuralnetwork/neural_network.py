import numpy as np
import math
import random
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

        for i in range(1, epochs):

            # shuffle the dataset on each iteration
            shuffled_X, shuffled_Y = self.shuffle_dataset(X, Y, train_size)

            # train each mini_batch
            for b in range(0, train_size, batch_size):

                # get the (next) mini_batch
                batch_A, batch_Y = self.get_mini_batch(shuffled_X, shuffled_Y, b, batch_size)

                # Forward Propagation
                for layer in self.layers:
                    batch_A = layer.forward_pass(batch_A)

                deriv_activation = self.loss.deriv(batch_Y, batch_A)

                # Backward propagation
                for layer in reversed(self.layers):
                    deriv_activation = layer.backward_pass(deriv_activation, learning_rate, train_size)


            print("cost after", i, "iterations:", self.cost(self.loss.fct(batch_Y, batch_A), train_size))


    def shuffle_dataset(self, X, Y, train_size):
        permutation = list(np.random.permutation(train_size))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]
        return shuffled_X, shuffled_Y


    def get_mini_batch(self, X, Y, pos, batch_size):
        batch_x = X[:, pos : pos + batch_size]
        batch_y = Y[:, pos : pos + batch_size]
        return batch_x, batch_y

    def predict(self, X):

        # first activation is our training set
        A = X

        # Number of classes to predict
        #
        for layer in self.layers:
            A = layer.forward_pass(A)

        return A
