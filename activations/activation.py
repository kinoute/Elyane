""" Abstract Class / Interface of our Activation Functions """


class Activation:

    """ This class is a interface that defines what the activation functions classes should contain. """

    def activ(self, F):
        """ The activation function formula.

        Args:
            F (array): Linear combinaison, most like W.X + b.

        Raises:
            NotImplementedError: In case the function has not been implemented.
        """

        raise NotImplementedError

    def deriv(self, F):
        """ The derivative of the activation function.

        Args:
            F (array): The derivative of the activation function according to the last activation output.

        Raises:
            NotImplementedError: In case the function has not been implemented.
        """

        raise NotImplementedError

    def heuristic(self, F):
        """ The heuristic formula to initialize our weights better depending of the activation function.

        Args:
            F (array): The heuristic formula.

        Raises:
            NotImplementedError: In case the function has not been implemented.
        """

        raise NotImplementedError
