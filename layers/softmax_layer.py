from .fc_layer import FCLayer
import numpy as np

class SoftmaxLayer(FCLayer):

    def backward_pass(self, deriv_activation, learning_rate, train_size):
        deriv_pre_activation = deriv_activation
        deriv_activation = np.dot(self.weights.T, deriv_pre_activation)

        # derivatives
        self.dWeights = np.dot(deriv_pre_activation, self.input.T) / train_size
        self.dBias = np.sum(deriv_pre_activation, axis = 1, keepdims = True) / train_size

        # update parameters
        self.weights -= learning_rate * self.dWeights
        self.bias -= learning_rate * self.dBias

        return deriv_activation
