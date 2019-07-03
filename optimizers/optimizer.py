""" The interface for the Optimizers classes """


class Optimizer:

    """ The abstract class / interface for the optimizers """

    def __init__(self):
        """ Initialize the optimizer.

        Raises:
            NotImplementedError: In case the function has not been implemented.
        """

        raise NotImplementedError

    def for_dw(self, deriv_weights):
        """ Get the optimized version of the weights derivative.

        Args:
            deriv_weights (array): The derivative of our weights.

        Raises:
            NotImplementedError: In case the function has not been implemented.
        """

        raise NotImplementedError

    def for_db(self, deriv_bias):
        """ Get the optimized version of the bias derivative.

        Args:
            deriv_bias (array): The derivative of our bias.

        Raises:
            NotImplementedError: In case the function has not been implemented.
        """

        raise NotImplementedError
