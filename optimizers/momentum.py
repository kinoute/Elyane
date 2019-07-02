""" The Momentum Optimizer """
from .optimizer import Optimizer


class Momentum(Optimizer):

    """ This class contains everything needed to use the Momentum Optimizer.

    Attributes:
        beta (float): Description
        var_db (array): Description
        var_dw (array): Description
    """

    def __init__(self):
        """ Initialize our attributes for the optimizer """

        self.var_dw = 0
        self.var_db = 0
        self.beta = 0.9

    def for_dw(self, deriv_weights):
        """ Initialize the default values for the Momentum Optimization

        Args:
            deriv_weights (array): The derivative of our weights.

        Returns:
            array: The variance of the derivative of our weights.
        """

        self.var_dw = (self.beta * self.var_dw) + (0.1 * deriv_weights)

        return self.var_dw

    def for_db(self, deriv_bias):
        """Optimize the derivative of our bias according to Momentum.

        Args:
            deriv_bias (array): Derivative of our bias.

        Returns:
            array: Returns Momentum-Optimized derivative of bias.
        """
        self.var_db = (self.beta * self.var_db) + (0.1 * deriv_bias)

        return self.var_db
