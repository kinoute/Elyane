""" Full Connected Layer Class. """
import numpy as np
from .layer import Layer
#from .optimizers import no_optim


class FCLayer(Layer):

    """ This class contains everything related to FC Layer.

    Attributes:
        activation (object): Instance of the activation function of the layer.
        activation_output (array): Output of the activation function.
        bias (array): Bias of the layer, depends of the input and ouput size of the layer.
        dBias (array): Derivative of the bias.
        dWeights (array): Derivative of the weights.
        input (array): The input of our layer. Can be the training set or output of the last layer.
        optim (object): Instance of the optimizer class picked for the network.
        pre_activation (array): Linear combinaison before the activation function.
        weights (array): The weights for this particular layer.
    """

    def __init__(self, input_size, output_size, activation, optimizer):
        """ Initialize our FC Layer.

        Args:
            input_size (int): Size of what comes into the layer.
            output_size (int): Size of what comes out of the layer.
            activation (object): Instance of the activation function class picked for this layer.
            optimizer (object): Instance of the optimizer class picked for the neural network.
        """

        self.activation = activation
        self.optimizer = optimizer

        # initialization of our weights and bias
        self.weights = np.random.randn(output_size, input_size) * self.activation.heuristic(input_size)
        self.bias = np.ones((output_size, 1))

    def forward_pass(self, input_data):
        """ The forward propagation features for the layer.

        Args:
            input_data (array): The input data for the forward can be the training data or the \
            ouput of the last layer.

        Returns:
            array: Returns the activation function output of the layer.
        """
        self.input = input_data
        self.pre_activation = np.dot(self.weights, self.input) + self.bias
        self.activation_output = self.activation.activ(self.pre_activation)

        return self.activation_output

    def backward_pass(self, deriv_activation, learning_rate, train_size):
        """ The backward propagation features for the layer.

        Args:
            deriv_activation (array): The Gradient of our loss function.
            learning_rate (float): The learning rate of the neural network.
            train_size (float): Number of samples in our training set. Can be equal to batch size.

        Returns:
            array: Returns the derivative of the activation function after updating parameters.
        """

        deriv_pre_activation = self.activation.deriv(self.activation_output) * deriv_activation
        deriv_activation = np.dot(self.weights.T, deriv_pre_activation)

        # derivatives
        self.deriv_weights = np.dot(deriv_pre_activation, self.input.T) / train_size
        self.deriv_bias = np.sum(deriv_pre_activation, axis=1, keepdims=True) / train_size

        # update parameters
        self.weights -= learning_rate * self.optimizer.for_dw(self.deriv_weights)
        self.bias -= learning_rate * self.optimizer.for_db(self.deriv_bias)

        return deriv_activation
