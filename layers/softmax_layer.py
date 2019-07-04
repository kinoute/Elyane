""" The Softmax Layer class. """
import numpy as np
from .fc_layer import FCLayer


class SoftmaxLayer(FCLayer):

    """ The SoftMax Layer Class lets us configure the backward specially for this type of activation.

    Attributes:
        deriv_bias (array): The derivative of our bias.
        deriv_weights (array): The derivative of our weights.
    """

    def backward_pass(self, deriv_activation, learning_rate, train_size, regularizer):
        """ The backward propagation for the softmax layer.

        Args:
            deriv_activation (array): The derivative of our activation function.
            learning_rate (float): The learning rate of our neural network.
            train_size (int): The number of examples of the training set.
            regularizer (object) : Instance of the regularizer class picked for the network.

        Returns:
            array: The derivative of the activation function
        """

        deriv_pre_activation = deriv_activation
        deriv_activation = np.dot(self.weights.T, deriv_pre_activation)

        # derivatives
        self.deriv_weights = np.dot(deriv_pre_activation, self.input.T) / train_size
        self.deriv_bias = np.sum(deriv_pre_activation, axis=1, keepdims=True) / train_size

        # update parameters
        self.weights -= learning_rate * (self.optimizer.for_dw(self.deriv_weights)
                                         + regularizer.backward(self.weights) / train_size)
        self.bias -= learning_rate * self.optimizer.for_db(self.deriv_bias)

        return deriv_activation
