"""Summary
"""


class Regularizer:
    """docstring for Regularizer
    """

    def __init__(self, lambd):
        """Summary

        Args:
            lambd (TYPE): Description

        Returns:
            TYPE: Description
        """
        return NotImplementedError

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
