""" The default optimizer when none is chosen """
from .optimizer import Optimizer


class NoOptim(Optimizer):

    """ The default Optimizer if none is chosen by the user. It's basicallly returns the values. """

    def for_dw(self, deriv_weights, time_step):
        """ Returns the deriv_weights directly since we don't have do to any calculation.

        Args:
            deriv_weights (array): The derivative of our weights.
            time_step (int): The time step of our neural network.

        Returns:
            array: The derivative of our weights.
        """
        return deriv_weights

    def for_db(self, deriv_bias, time_step):
        """ Returns the deriv_bias directly since we don't have do to any calculation.

        Args:
            deriv_bias (array): The derivative of our bias.
            time_step (int): The time step of our neural network.

        Returns:
            array: The derivative of our bias.
        """
        return deriv_bias
