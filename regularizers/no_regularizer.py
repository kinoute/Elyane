import numpy as np
from .regularizer import Regularizer


class NoReg(Regularizer):
    """docstring for NoReg"""

    def __init__(self, lambd):

        self.l2_cost = 0
        self.lambd = lambd

    def l2_forward(self, weights):
        return self.l2_cost

    def l2_backward(self, weights):
        return self.l2_cost
