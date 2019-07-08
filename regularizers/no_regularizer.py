""" The No Regularizer Class when no regularizer is picked by the user """

import numpy as np
from .regularizer import Regularizer


class NoReg(Regularizer):

    """ The NoReg class is called when no Regularizer is defined.

    Attributes:
        lambd (float, optional): The hyper-parameter lambda for the L1 or L2 regularization.
    """

    def forward(self, weights):
        """ The forward stage for the regularizer. Here, we return zero as there is no regularizer.

        Args:
            weights (array): The weights of our layer.

        Returns:
            int: Returns 0 to cancel any regularization work.
        """

        return 0

    def backward(self, weights):
        """ The backward stage for the regularizer. Here we return zero as there is no regularizer picked by the user.
        Args:
            weights (array): The weights of our layer.

        Returns:
            int: Returns 0 to cancel any regularization formula.
        """

        return 0
