""" Adam Optimization without bias correction """

import numpy as np
from .optimizer import Optimizer


class Adam(Optimizer):
    """ Class for the Adam Optimization algorithm.

    The Adam optimizer with default values and without bias correction.

    Extends:
        Optimization

    Attributes:
        beta1 (float): The hyper-parameter for the momentum algorithm.
        beta2 (float): The hyper-parameter for the RMSProp algorithm.
        epsilon (float): A small value to avoid divide by zero errors.
        mean_db (array): Derivative of the bias with.
        mean_dw (array): Derivative of the weights with.
        var_dw (array): Variance of the weights derivative.
        var_db (array): Variance of the bias derivative.
    """

    def __init__(self):
        """ Initialize the default values for the Adam Optimization """

        self.var_dw = 0
        self.var_db = 0
        self.mean_dw = 0
        self.mean_db = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    def for_dw(self, deriv_weights, time_step):
        """Optimize the derivative of our weights.

        Args:
            deriv_weights (array): Derivative of our weights.

        Returns:
            array: Returns Adam-Optimized derivative of weights.
        """

        # momentum and rmsprop calculation
        self.var_dw = (self.beta1 * self.var_dw) + ((1 - self.beta1) * deriv_weights)
        self.mean_dw = (self.beta2 * self.mean_dw) + ((1 - self.beta2) * np.square(deriv_weights))

        # bias correction
        # self.var_dw = self.var_dw / (1 - np.power(self.beta1, t))
        # self.mean_dw = self.mean_dw / (1 - np.power(self.beta2, t))

        return self.var_dw / (np.sqrt(self.mean_dw) + self.epsilon)

    def for_db(self, deriv_bias, time_step):
        """Optimize the derivative of our bias.

        Args:
            deriv_bias (array): Derivative of our bias.

        Returns:
            array: Returns Adam-Optimized derivative of bias.
        """

        # momentum and rmsprop calculation
        self.var_db = (self.beta1 * self.var_db) + ((1 - self.beta1) * deriv_bias)
        self.mean_db = (self.beta2 * self.mean_db) + ((1 - self.beta2) * np.square(deriv_bias))

        # bias correction
        # self.var_db /= (1 - np.power(self.beta1, t))
        # self.mean_db /= (1 - np.power(self.beta2, t))

        return self.var_db / (np.sqrt(self.mean_db) + self.epsilon)
