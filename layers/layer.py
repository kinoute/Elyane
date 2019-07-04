""" The Abstract Class / Interface for our Layers """


class Layer:

    """ This class shows what functions are mandatory for each layer class. """

    def __init__(self, input_size, output_size, activation):
        """ What our abstract Layer class needs to have for initialization.

        Args:
            input_size (int): The input size of our layer.
            output_size (int): The output size of our layer.
            activation (object): The activation function used for this layer.

        Raises:
            NotImplementedError: In case the function has not been implemented.
        """

        raise NotImplementedError

    def forward_pass(self, input_data):
        """ The forward propagation features for our layers.

        Args:
            input_data (array): The input data for the forward can be the training data or the \
            ouput of the last layer.

        Raises:
            NotImplementedError: In case the function has not been implemented.
        """

        raise NotImplementedError

    def backward_pass(self, deriv_activation, learning_rate, train_size, regularizer):
        """ The backward propagation features for the layer.

        Args:
            deriv_activation (array): The Gradient of our loss function.
            learning_rate (float): The learning rate of the neural network.
            train_size (float): Number of samples in our training set. Can be equal to batch size.
            regularizer (object) : Instance of the regularizer class picked for the network.

        Raises:
            NotImplementedError: In case the function has not been implemented.
        """

        raise NotImplementedError
