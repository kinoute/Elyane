
class Regularizer:
    """docstring for Regularizer"""

    def __init__(self, lambd):
        return NotImplementedError

    def l2_forward(self, weights):
        return NotImplementedError

    def l2_backward(self, weights):
        return NotImplementedError
