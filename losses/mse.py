import numpy as np
from .loss import Loss

class MSE(Loss):

    def fct(self, Y, A):
        return np.mean(np.power(Y - A, 2))

    def deriv(self, Y, A):
        return 2 * (A - Y) / Y.size
