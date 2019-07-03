""" The Multi-Class Cross-Entropy Loss Function """
import numpy as np
from .loss import Loss


class MultiClassCrossEntropy(Loss):

    """ The Multi-class CrossEntropy that contains everything needed to calculate the loss. """

    def fct(self, labels, preds):
        """ The loss function formula.

        Args:
            labels (array): The true labels of our dataset.
            preds (array): The predictions made by our neural networ.

        Returns:
            array: Returns the loss according to A and Y.
        """

        return - np.sum(labels * np.log(preds))

    def deriv(self, labels, preds):
        """ Calculates the derivative of the CrossEntropy loss function.

        Args:
            labels (array): The true labels of our dataset.
            preds (array): The predictions made by our neural network

        Returns:
            array: Returns the derivative loss according to A and Y.
        """

        return preds - labels
