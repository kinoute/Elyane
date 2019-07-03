""" The TanH activation function """
import numpy as np
from .activation import Activation


class TanH(Activation):

    """ The TanH activation function Class to handle multi class classification. """

    def activ(self, data):
        """ The activation function formula for the TanH.

        Args:
            data (array): Linear combinaison, most likely W.X + b.

        Returns:
            array: Returns the result of the TanH activation function.
        """

        return np.tanh(data)

    def deriv(self, data):
        """ The derivative of the TanH activation function.

        Args:
            data (array): The derivative of the TanH function according to the last activation output.

        Returns:
            array: Returns the result of the derivative of the TanH according to data.
        """

        return 1 - np.square(data)

    def heuristic(self, data):
        """ The heuristic formula to initialize our weights better when using the TanH.

        Args:
            data (float): The size of the layer(s).

        Returns:
            array: Returns the heuristic for TanH to initialize layer's weights better.
        """

        return np.sqrt(1 / data)
