""" The L1 Regularization Class """

import numpy as np
from .regularizer import Regularizer


class L1Regularizer(Regularizer):

    """ The L1 Regularizer Class to reduce overfitting.

    Attributes:
        lambd (float, optional): The hyper-parameter lambda for the L1 or L2 regularization.
    """

    def forward(self, weights):
        """ The forward stage for the L1 regularization.

        Args:
            weights (array): The weights of our layer.

        Returns:
            array: Returns the regularized weights of our layer to compute cost.
        """

        return (np.sum(np.abs(weights))) * (self.lambd / 2)

    def backward(self, weights):
        """ The backward stage for the L1 regularization.

        Args:
            weights (array): The weights of our layer.

        Returns:
            array: Returns the regularized weights of our layer to update parameters.
        """

        mask_1 = (weights >= 0) * 1.0
        mask_2 = (weights < 0) * -1.0

        return (self.lambd * (mask_1 + mask_2)) / 2
