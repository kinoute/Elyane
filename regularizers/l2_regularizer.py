""" The L2 Regularization Class """
import numpy as np
from .regularizer import Regularizer


class L2Regularizer(Regularizer):

    """ The L2 Regularization class to reduce overfitting.

    Attributes:
        cost (int): Description
        lambd (float): The hyper parameter for the L2 Regularization.
    """

    def forward(self, weights):
        """Summary

        Args:
            weights (TYPE): Description

        Returns:
            TYPE: Description
        """

        return self.lambd * (np.sum(np.square(weights)))

        return self.cost

    def backward(self, weights):
        """Summary

        Args:
            weights (TYPE): Description

        Returns:
            TYPE: Description
        """

        return self.lambd * weights
