""" The Leaky Relu activation function"""
import numpy as np
from activations import Activation


class LeakyRelu(Activation):

    """ The Leaky Relu activation function Class. """

    def activ(self, F):
        """Summary

        Args:
            F (TYPE): Description

        Returns:
            TYPE: Description
        """
        return np.where(F > 0, F, F * 0.01)

    def deriv(self, F):
        """Summary

        Args:
            F (TYPE): Description

        Returns:
            TYPE: Description
        """
        return np.clip(F > 0, 0.01, 1.0)

    def heuristic(self, F):
        """Summary

        Args:
            F (TYPE): Description

        Returns:
            TYPE: Description
        """
        return np.sqrt(2 / F)
