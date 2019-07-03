""" Abstract Class / Interface of our Activation Functions """


class Activation:

    """ This class is a interface that defines what the activation functions classes should contain. """

    def activ(self, data):
        """ The activation function formula.

        Args:
            data (array): Linear combinaison, most like W.X + b.

        Raises:
            NotImplementedError: In case the function has not been implemented.
        """

        raise NotImplementedError

    def deriv(self, data):
        """ The derivative of the activation function.

        Args:
            data (array): The derivative of the activation function according to the last activation output.

        Raises:
            NotImplementedError: In case the function has not been implemented.
        """

        raise NotImplementedError

    def heuristic(self, data):
        """ The heuristic formula to initialize our weights better depending of the activation function.

        Args:
            data (array): The heuristic formula.

        Raises:
            NotImplementedError: In case the function has not been implemented.
        """

        raise NotImplementedError
