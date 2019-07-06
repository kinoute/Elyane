""" The Regularizer Abstract Class / Interface """


class Regularizer:

    """ The Regularizer Abstract Class to reduce overfitting.

    Attributes:
        lambd (float, optional): The hyper-parameter lambda for the L1 or L2 regularization.
    """

    def __init__(self, lambd=0):
        """ Initialize our Regularizer.

        Args:
            lambd (float, optional): The hyper-parameter lambda for the L1 or L2 regularization.
        """

        self.lambd = lambd

    def forward(self, weights):
        """ The forward stage for the regularizer.

        Args:
            weights (array): The weights of our layer.

        Returns:
            array: Returns the regularized weights of our layer to compute cost.
        """

        return NotImplementedError

    def backward(self, weights):
        """ The backward stage for the regularizer.

        Args:
            weights (array): The weights of our layer.

        Returns:
            array: Returns the regularized weights of our layer to update parameters.
        """

        return NotImplementedError
