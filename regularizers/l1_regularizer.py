"""Summary
"""
import numpy as np
from .regularizer import Regularizer


class L1Regularizer(Regularizer):

    """Summary

    Attributes:
        cost (int): Description
        lambd (TYPE): Description
    """

    def __init__(self, lambd=0):
        """Summary

        Args:
            lambd (int, optional): Description
        """
        self.cost = 0
        self.lambd = lambd

    def forward(self, weights):
        """Summary

        Args:
            weights (TYPE): Description

        Returns:
            TYPE: Description
        """
        self.cost += (np.sum(np.abs(weights))) * (self.lambd / 2)

        return self.cost

    def backward(self, weights):
        """Summary

        Args:
            weights (TYPE): Description

        Returns:
            TYPE: Description
        """
        mask_1 = (weights >= 0) * 1.0
        mask_2 = (weights < 0) * -1.0

        return (self.lambd * (mask_1 + mask_2) / 2
