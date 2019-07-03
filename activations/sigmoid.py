""" The Sigmoid activation function """
import numpy as np
from .activation import Activation


class Sigmoid(Activation):

    """ The Sigmoid activation function Class. """

    def activ(self, F):
        """ The activation function formula for the Sigmoid.

        Args:
            F (array): Linear combinaison, most likely W.X + b.

        Returns:
            array: Returns the result of the Sigmoid activation function.
        """

        return 1 / (1 + np.exp(- F))

    def deriv(self, F):
        """ The derivative of the Sigmoid activation function.

        Args:
            F (array): The derivative of the Sigmoid function according to the last activation output.

        Returns:
            array: Returns the result of the derivative of the Sigmoid according to F.
        """

        return F * (1 - F)

    def heuristic(self, F):
        """ The heuristic formula to initialize our weights better when using the Sigmoid.

        Args:
            F (float): The size of the layer(s).

        Returns:
            array: Returns the heuristic for Sigmoid to initialize layer's weights better.
        """

        return np.sqrt(1 / F)
