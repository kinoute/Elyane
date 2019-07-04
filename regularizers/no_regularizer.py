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
        return self.cost

    def backward(self, weights):
        """Summary

        Args:
            weights (TYPE): Description

        Returns:
            TYPE: Description
        """
        return self.cost
