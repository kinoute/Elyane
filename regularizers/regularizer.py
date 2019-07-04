"""Summary
"""


class Regularizer:
    """docstring for Regularizer
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
        return NotImplementedError

    def backward(self, weights):
        """Summary

        Args:
            weights (TYPE): Description

        Returns:
            TYPE: Description
        """
        return NotImplementedError
