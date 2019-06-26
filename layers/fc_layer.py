from .layer import Layer
import numpy as np

class FCLayer(Layer):

    def __init__(self, input_size, output_size, activation):
        self.activation = activation
        self.weights = np.random.randn(output_size, input_size) * self.activation.heuristic(input_size)
        self.bias = np.ones((output_size, 1))

    def forward_pass(self, input_data):
        self.input = input_data
        self.pre_activation = np.dot(self.weights, self.input) + self.bias
        self.activation_output = self.activation.activ(self.pre_activation)

        return self.activation_output

    def backward_pass(self, deriv_activation, learning_rate, train_size):
        deriv_pre_activation = self.activation.deriv(self.activation_output) * deriv_activation
        deriv_activation = np.dot(self.weights.T, deriv_pre_activation)

        # derivatives
        self.dWeights = np.dot(deriv_pre_activation, self.input.T) / train_size
        self.dBias = np.sum(deriv_pre_activation ,axis = 1, keepdims = True) / train_size

        # update parameters
        self.weights -= learning_rate * self.dWeights
        self.bias -= learning_rate * self.dBias

        return deriv_activation
