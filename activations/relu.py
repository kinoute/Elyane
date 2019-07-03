""" The Relu activation function """
from .activation import Activation
import numpy as np


class Relu(Activation):

    """ The Relu activation function Class. """

    def activ(self, F):
        """ The activation function formula for the Relu.

        Args:
            F (array): Linear combinaison, most likely W.X + b.

        Returns:
            array: Returns the result of the Relu activation function.
        """

        return np.maximum(0, F)

    def deriv(self, F):
        """ The derivative of the Relu activation function.

        Args:
            F (array): The derivative of the Relu function according to the last activation output.

        Returns:
            array: Returns the result of the derivative of the Relu according to F.
        """

        return (F > 0).astype(float)

    def heuristic(self, F):
        """ The heuristic formula to initialize our weights better when using the Relu.

        Args:
            F (float): The size of the layer(s).

        Returns:
            array: Returns the heuristic for Relu to initialize layer's weights better.
        """

        return np.sqrt(2 / F)
