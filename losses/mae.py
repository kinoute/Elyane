""" The Mean Absolute Error Loss Function """
import numpy as np
from .loss import Loss


class MAE(Loss):

    """ The MAE Class that contains everything needed to calculate the loss. """

    def fct(self, labels, preds):
        """ The MAE loss function formula.

        Args:
            labels (array): The true labels of our dataset.
            preds (array): The predictions made by our neural network.

        Returns:
            array: Returns the loss according to A and Y.
        """

        return np.mean(np.abs(preds - labels))

    def deriv(self, labels, preds):
        """ Calculates the derivative of the MAE loss function.

        Args:
            Y (array): The true labels of our dataset.
            A (array): The predictions made by our neural network.

        Returns:
            array: Returns the derivative loss according to A and Y.
        """

        return np.where(preds > labels, 1., -1.)
