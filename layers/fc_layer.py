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

    def backward_pass(self, loss, output_error, learning_rate, m):
        #input_error = np.dot(dZ, self.weights.T)
        #self.dZ = loss.deriv(Y, self.activation_output) * self.activation.deriv(self.activation_output)
        #self.dZ = np.dot(self.weights, dZ) * self.activation.deriv(self.input)
        dZ = self.activation.deriv(self.activation_output) * output_error
        output_error = np.dot(self.weights.T, dZ)
        self.dWeights = np.dot(dZ, self.input.T) / m
        self.dBias = np.sum(dZ ,axis = 1, keepdims = True) / m

        # update parameters
        self.weights -= learning_rate * self.dWeights
        self.bias -= learning_rate * self.dBias

        return output_error
