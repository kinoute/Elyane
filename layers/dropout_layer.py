""" The DropOut Layer Class. """
from .layer import Layer
import numpy as np


class DropOut(Layer):
    """ This class contains everything related to DropOut Layer.

    Attributes:
        dropout (array): Mask of the activation function's output with the given possibility rate.
        input (array): The output of the previous layer's forward_pass.
        output (array): Masked state of the input.
        rate (float): The probability given by the user to drop hidden-units in the previous layer.
    """

    def __init__(self, rate):
        """ Initialisation for the dropout layer.

        Args:
            rate (float): The probability given by the user to drop hidden-units in the previous layer.
        """

        self.rate = 1 - rate

    def forward_pass(self, input_data):
        """ The forward propagation features for the layer.

        Args:
            input (array): The output of the previous layer's forward_pass.

        Returns:
            output (array): Masked state of the input for forward propagation.
        """

        self.input = input_data
        self.dropout = np.random.rand(self.input.shape[0], self.input.shape[1])
        self.dropout = (self.dropout < self.rate)
        self.output = self.input * self.dropout
        self.output /= self.rate
        return self.output

    def backward_pass(self, deriv_activation, learning_rate, train_size):
        """ The backward propagation features for the layer.

        Args:
            deriv_activation (array): The Gradient of our loss function.
            learning_rate (float): The learning rate of the neural network.
            train_size (float): Number of samples in our training set. Can be equal to batch size.

        Returns:
            output (array): Masked state of the deriv_activation for backward propagation.
        """

        self.output = deriv_activation
        self.output *= self.dropout
        self.output /= self.rate
        return self.output
