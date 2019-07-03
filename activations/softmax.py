""" The Softmax activation function """

import numpy as np
from .activation import Activation


class Softmax(Activation):

    """ The Sigmoid activation function Class to handle multi class classification. """

    def activ(self, data):
        """ The activation function formula for the Softmax.

        Args:
            data (array): Linear combinaison, most likely W.X + b.

        Returns:
            array: Returns the result of the Softmax activation function.
        """

        # return np.exp(data)/np.sum(np.exp(data), axis=0)
        maxVal = np.max(data, axis=0, keepdims=True)  # To normalize the values for numerical stability
        return np.exp(data - maxVal) / np.sum(np.exp(data - maxVal), axis=0, keepdims=True)

    def deriv(self, data):
        """ The derivative of the Softmax activation function.

        Args:
            data (array): The derivative of the Softmax function according to the last activation output.

        Returns:
            array: Returns the result of the derivative of the Softmax according to data.
        """

        return data * (1 - data)
        #deriv = data.reshape(-1, 1)
        # return np.diagflat(deriv) - np.dot(data, deriv.T)

    def heuristic(self, data):
        """ The heuristic formula to initialize our weights better when using the Softmax.

        Args:
            data (float): The size of the layer(s).

        Returns:
            array: Returns the heuristic for Softmax to initialize layer's weights better.
        """

        return np.sqrt(1 / data)
