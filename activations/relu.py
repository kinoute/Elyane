""" The Relu activation function """
from .activation import Activation
import numpy as np


class Relu(Activation):

    """ The Relu activation function Class. """

    def activ(self, data):
        """ The activation function formula for the Relu.

        Args:
            data (array): Linear combinaison, most likely W.X + b.

        Returns:
            array: Returns the result of the Relu activation function.
        """

        return np.maximum(0, data)

    def deriv(self, data):
        """ The derivative of the Relu activation function.

        Args:
            data (array): The derivative of the Relu function according to the last activation output.

        Returns:
            array: Returns the result of the derivative of the Relu according to data.
        """

        return (data > 0).astype(float)

    def heuristic(self, data):
        """ The heuristic formula to initialize our weights better when using the Relu.

        Args:
            data (float): The size of the layer(s).

        Returns:
            array: Returns the heuristic for Relu to initialize layer's weights better.
        """

        return np.sqrt(2 / data)
