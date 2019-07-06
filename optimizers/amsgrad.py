""" AmsGrad Optimization without bias correction """

import numpy as np
from .adam import Adam


class AmsGrad(Adam):
    """ Class for the AmsGrad Optimization algorithm.

    The AmsGrad optimizer with default values and without bias correction.

    Extends:
        Adam

    Attributes:
        var_dw (array): Variance of the weights derivative.
        var_db (array): Variance of the bias derivative.
        mean_dw (array): Derivative of the weights with.
        mean_db (array): Derivative of the bias with.
        beta1 (float): The hyper-parameter for the momentum algorithm.
        beta2 (float): The hyper-parameter for the RMSProp algorithm.
        epsilon (float): A small value to avoid divide by zero errors.
    """

    def for_dw(self, deriv_weights):
        """ Optimize the derivative of our weights.

        Args:
            deriv_weights (array): Derivative of our weights.

        Returns:
            array: Returns AmsGrad-Optimized derivative of weights.
        """

        # momentum and rmsprop calculation
        self.var_dw = (self.beta1 * self.var_dw) + ((1 - self.beta1) * deriv_weights)
        # correct adam convergence problem
        meandw = (self.beta2 * self.mean_dw) + ((1 - self.beta2) * np.square(deriv_weights))
        self.mean_dw = np.maximum(meandw, self.mean_dw)

        # bias correction
        # self.var_dw = self.var_dw / (1 - np.power(self.beta1, t))
        # self.mean_dw = self.mean_dw / (1 - np.power(self.beta2, t))

        return self.var_dw / (np.sqrt(self.mean_dw) + self.epsilon)
