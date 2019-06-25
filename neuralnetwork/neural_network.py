import numpy as np
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None
    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # add loss function to network
    def use(self, loss):
        self.loss = loss

    def cost(self, loss, size):
        return (np.sum(loss) / size)

    def train(self, X, Y, epochs, learning_rate):
        m = X.shape[1]

        for i in range(1000):
            A = X # AO equals to the training  set

            for layer in self.layers:
                A = layer.forward_pass(A)

            print("cost:", self.cost(self.loss.fct(Y,A), m))

            output_error = A - Y

            for i, layer in enumerate(reversed(self.layers)):
                output_error = layer.backward_pass(self.loss, output_error, learning_rate, m)





