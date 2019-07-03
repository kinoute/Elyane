""" The CrossEntropy Loss Function """
import numpy as np
from .loss import Loss


class CrossEntropy(Loss):

    """ The CrossEntropy Class that contains everything needed to calculate the loss. """

    def fct(self, labels, preds):
        """ The loss function formula.

        Args:
            labels (array): The true labels of our dataset.
            preds (array): The predictions made by our neural networ.

        Returns:
            array: Returns the loss according to A and Y.
        """

        return - (labels * np.log(preds) + (1 - labels) * np.log(1 - preds))

    def deriv(self, labels, preds):
        """ Calculates the derivative of the CrossEntropy loss function.

        Args:
            labels (array): The true labels of our dataset.
            preds (array): The predictions made by our neural network

        Returns:
            array: Returns the derivative loss according to A and Y.
        """

        return - (np.divide(labels, preds) - np.divide(1 - labels, 1 - preds))
