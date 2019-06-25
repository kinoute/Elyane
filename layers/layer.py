class Layer(object):

    def __init__(self, input_size, output_size, activation):
        raise NotImplementedError

    def forward_pass(self, input_data):
        raise NotImplementedError

    def backward_pass(self, output, learning_rate):
        raise NotImplementedError
