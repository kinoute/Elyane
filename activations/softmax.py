""" The Softmax activation function """

import numpy as np
from .activation import Activation


class Softmax(Activation):

    """ The Sigmoid activation function Class to handle multi class classification. """

    def activ(self, F):
        """ The activation function formula for the Softmax.

        Args:
            F (array): Linear combinaison, most likely W.X + b.

        Returns:
            array: Returns the result of the Softmax activation function.
        """

        # return np.exp(F)/np.sum(np.exp(F), axis=0)
        maxVal = np.max(F, axis=0, keepdims=True)  # To normalize the values for numerical stability
        return np.exp(F - maxVal) / np.sum(np.exp(F - maxVal), axis=0, keepdims=True)

    def deriv(self, F):
        """ The derivative of the Softmax activation function.

        Args:
            F (array): The derivative of the Softmax function according to the last activation output.

        Returns:
            array: Returns the result of the derivative of the Softmax according to F.
        """

        return F * (1 - F)
        #deriv = F.reshape(-1, 1)
        # return np.diagflat(deriv) - np.dot(F, deriv.T)

    def heuristic(self, F):
        """ The heuristic formula to initialize our weights better when using the Softmax.

        Args:
            F (float): The size of the layer(s).

        Returns:
            array: Returns the heuristic for Softmax to initialize layer's weights better.
        """

        return np.sqrt(1 / F)
