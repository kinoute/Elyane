""" The Neural Network class. """
import numpy as np


class NeuralNetwork:

    """ The Neural Network class that handles everything about the training.

    Attributes:
        layers (list): List that contains the instances of the NN's layers.
        loss (object): Object that contains the instance of the picked loss function class.
    """

    def __init__(self):
        """ Initialize our neural network. """

        self.layers = []
        self.loss = None
        self.regularizer = None
        self.weights = 0

    def add(self, layer):
        """ Adds a layer to our neural network's structure.

        Args:
            layer (object): Instance of the picked layer class.
        """

        self.layers.append(layer)

    def use(self, loss, regularizer):
        """ Defines the loss function that will be used for the entire neural network.

        Args:
            loss (object): Instance of the picked loss function class.
        """

        self.loss = loss
        self.regularizer = regularizer

    def cost(self, loss, size):
        """ The cost function formula to check the training status.

        Args:
            loss (array): The loss of the training set after each iteration.
            size (float): Size of the training set.

        Returns:
            float: Returns the cost value of the neural network.
        """

        return np.sum(loss) / size

    def train(self, x_train, y_train, epochs, learning_rate, batch_size=128):
        """ Starts the training part of our neural network.

        Args:
            x_train (array): The dataset that will be used to train our neural network.
            y_train (array): The labels of the dataset, i.e the good answers.
            epochs (int): The number of iterations will do to train the neural network.
            learning_rate (float): The speed at how fast our model will learn.
            batch_size (int, optional): size of each mini-batch. Default: 128.
        """

        # Number of samples in the entire training set
        train_size = x_train.shape[1]

        for i in range(1, epochs + 1):

            # shuffle the dataset on each iteration
            shuffled_x, shuffled_y = self.shuffle_dataset(x_train, y_train, train_size)

            # train each mini_batch
            for batch in range(0, train_size, batch_size):

                # get the (next) mini_batch
                batch_a, batch_y = self.get_mini_batch(shuffled_x, shuffled_y, batch, batch_size)

                # forward propagation
                for layer in self.layers:
                    batch_a, weights = layer.forward_pass(batch_a)

                # compute cost
                cost = self.cost(self.loss.fct(batch_y, batch_a), batch_size) + (self.regularizer.forward(weights) / (2 * batch_size))

                deriv_activation = self.loss.deriv(batch_y, batch_a)

                # backward propagation
                for layer in reversed(self.layers):
                    deriv_activation = layer.backward_pass(deriv_activation,
                                                           learning_rate,
                                                           batch_size,
                                                           self.regularizer)

            print("cost after", i, "iterations:", cost)

    def shuffle_dataset(self, x_train, y_train, train_size):
        """ Shuffle our dataset for the mini batch gradient descent.

        Args:
            x_train (array): The dataset that will be used to train our neural network.
            y_train (array): The labels of the dataset, i.e the good answers.
            train_size (float): Size of the training set.

        Returns:
            array: Returns shuffled training dataset and shuffled training labels.
        """

        permutation = list(np.random.permutation(train_size))
        shuffled_x = x_train[:, permutation]
        shuffled_y = y_train[:, permutation]

        return shuffled_x, shuffled_y

    def get_mini_batch(self, x_train, y_train, pos, batch_size):
        """ Gets a mini batch according to a position in the original dataset.

        Args:
            x_train (array): The dataset that will be used to train our neural network.
            y_train (array): The labels of the dataset, i.e the good answers.
            pos (float): Position into the dataset to do the slicing.
            batch_size (int): Size of our mini batches.

        Returns:
            array: Returns mini batch for training examples and training labels.
        """

        batch_x = x_train[:, pos: pos + batch_size]
        batch_y = y_train[:, pos: pos + batch_size]

        return batch_x, batch_y

    def predict(self, data):
        """ Gets prediction for the given datas.

        Args:
            data (array): Datas we are trying to get predictions from.

        Returns:
            array: Returns the prediction of our neural network for the given datas.
        """

        activ = data

        for layer in self.layers:
            activ, _ = layer.forward_pass(activ)

        return activ
