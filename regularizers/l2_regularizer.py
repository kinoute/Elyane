""" The L2 Regularization Class """
import numpy as np
from .regularizer import Regularizer


class L2Regularizer(Regularizer):

    """ The L2 Regularization class to reduce overfitting.

    Attributes:
        l2_cost (int): Description
        lambd (float): The hyper parameter for the L2 Regularization.
    """

    def __init__(self, lambd):
        """Summary

        Args:
            lambd (TYPE): Description
        """
        self.l2_cost = 0
        self.lambd = lambd

    def l2_forward(self, weights):
        """Summary

        Args:
            weights (TYPE): Description

        Returns:
            TYPE: Description
        """
        self.l2_cost += (np.sum(np.square(weights))) * (self.lambd / 2)

        return self.l2_cost

    def l2_backward(self, weights):
        """Summary

        Args:
            weights (TYPE): Description

        Returns:
            TYPE: Description
        """
        return (self.lambd * weights)
