"""Summary
"""
import numpy as np
from .regularizer import Regularizer


class NoReg(Regularizer):

    """Summary

    Attributes:
        cost (int): Description
        lambd (TYPE): Description
    """

    def forward(self, weights):
        """Summary

        Args:
            weights (TYPE): Description

        Returns:
            TYPE: Description
        """
        return self.cost

    def backward(self, weights):
        """Summary

        Args:
            weights (TYPE): Description

        Returns:
            TYPE: Description
        """
        return self.cost
