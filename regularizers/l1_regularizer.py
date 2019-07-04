import numpy as np
from .regularizer import Regularizer


class L1Regularizer(Regularizer):
    """docstring for L1Regularizer"""

    def __init__(self, lambd):

        self.l2_cost = 0
        self.lambd = lambd

    def l1_forward(self, weights):

        self.l1_cost += (np.sum(np.abs(weights))) * (self.lambd / 2)

        return self.l1_cost

    def l1_backward(self, weights):

        mask_1 = (weights >= 0) * 1.0
        mask_2 = (weights < 0) * -1.0

        return (self.lambd * (mask_1 + mask_2)/2
