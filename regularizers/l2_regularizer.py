import numpy as np
from .regularizer import Regularizer


class L2Regularizer(Regularizer):
    """docstring for L2Regularizer"""

    def __init__(self, lambd):

        self.l2_cost = 0
        self.lambd = lambd

    def l2_forward(self, weights):

        self.l2_cost += (np.sum(np.square(weights))) * (self.lambd / 2)

        return self.l2_cost

    def l2_backward(self, weights):

        return (self.lambd * weights)
