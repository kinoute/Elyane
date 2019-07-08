""" The RMSprop optimization """

import numpy as np
from .optimizer import Optimizer


class RMSprop(Optimizer):

    """ This class contains everything related to the RMSprop Optimizer.

    Attributes:
        mean_dw (array): The mean of the derivative of our weights.
        mean_db (array): The mean of the derivative of our bias.
        beta (float): The hyper-parameter for the RMSProp algorithm.
        epsilon (float): A small value to avoid divide by zero errors.

    """

    def __init__(self):
        """ Initialize the RMSprop optimizer """

        self.mean_dw = 0
        self.mean_db = 0
        self.beta = 0.999
        self.epsilon = 1e-8

    def for_dw(self, deriv_weights):
        """ Optimize the derivative of our weights according to RMSprop.

        Args:
            deriv_weights (array): The derivative of our weights.

        Returns:
            array: Returns optimized derivative of our weights.
        """

        self.mean_dw = (self.beta * self.mean_dw) + ((1 - self.beta) * np.power(deriv_weights, 2))

        return deriv_weights / (np.sqrt(self.mean_dw) + self.epsilon)

    def for_db(self, deriv_bias):
        """ Optimize the derivative of our bias according to RMSprop.

        Args:
            deriv_bias (array): The derivative of our bias.

        Returns:
            array: Returns optimized derivative of our bias.
        """

        self.mean_db = (self.beta * self.mean_db) + ((1 - self.beta) * np.power(deriv_bias, 2))

        return deriv_bias / (np.sqrt(self.mean_db) + self.epsilon)
