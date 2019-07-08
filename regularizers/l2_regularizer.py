""" The L2 Regularization Class """

import numpy as np
from .regularizer import Regularizer


class L2Regularizer(Regularizer):

    """ The L2 Regularization class to reduce overfitting.

    Attributes:
        lambd (float, optional): The hyper-parameter lambda for the L1 or L2 regularization.
    """

    def forward(self, weights):
        """ The forward stage for the L2 regularization.

        Args:
            weights (array): The weights of our layer.

        Returns:
            array: Returns the regularized weights of our layer to compute cost.
        """

        return (np.sum(np.square(weights))) * (self.lambd / 2)

    def backward(self, weights):
        """ The backward stage for the L2 regularization.

        Args:
            weights (array): The weights of our layer.

        Returns:
            array: Returns the regularized weights of our layer to update parameters.
        """

        return self.lambd * weights
