""" The Mean Squared Error Loss Function """

import numpy as np
from .loss import Loss


class MSE(Loss):

    """ The MSE Class that contains everything needed to calculate the loss. """

    def fct(self, labels, preds):
        """ The MSE loss function formula.

        Args:
            labels (array): The true labels of our dataset.
            preds (array): The predictions made by our neural network.

        Returns:
            array: Returns the loss according to A and Y.
        """

        return np.mean(np.power(labels - preds, 2))

    def deriv(self, labels, preds):
        """ Calculates the derivative of the MSE loss function.

        Args:
            labels (array): The true labels of our dataset.
            preds (array): The predictions made by our neural network.

        Returns:
            array: Returns the derivative loss according to A and Y.
        """

        return 2 * (preds - labels) / labels.size
