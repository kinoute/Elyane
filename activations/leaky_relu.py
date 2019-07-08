""" The Leaky Relu activation function"""

import numpy as np
from .activation import Activation


class LeakyRelu(Activation):

    """ The Leaky Relu activation function Class. """

    def activ(self, data):
        """ The activation function formula for the Leaky Relu.

        Args:
            data (array): Linear combinaison, most likely W.X + b.

        Returns:
            array: Returns the result of the Leaky Relu activation function.
        """

        return np.where(data > 0, data, data * 0.01)

    def deriv(self, data):
        """ The derivative of the Leaky Relu activation function.

        Args:
            data (array): The derivative of the activation function according to the last activation output.

        Returns:
            array: Returns the result of the derivative of the Leaky Relu according to data.
        """

        return np.clip(data > 0, 0.01, 1.0)

    def heuristic(self, data):
        """ The heuristic formula to initialize our weights better when using the Leaky Relu.

        Args:
            data (float): The size of the layer(s).

        Returns:
            array: Returns the heuristic for leaky Relu to initialize layer's weights better.
        """

        return np.sqrt(2 / data)
