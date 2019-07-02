""" The Abstract Class / Interface for our Layers
"""


class Layer:

    """ This class shows what functions are mandatory for each layer class.
    """

    def __init__(self, input_size, output_size, activation):
        """Summary

        Args:
            input_size (TYPE): Description
            output_size (TYPE): Description
            activation (TYPE): Description

        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError

    def forward_pass(self, input_data):
        """Summary

        Args:
            input_data (TYPE): Description

        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError

    def backward_pass(self, output, learning_rate):
        """Summary

        Args:
            output (TYPE): Description
            learning_rate (TYPE): Description

        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError
