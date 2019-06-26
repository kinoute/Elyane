import numpy as np
from .loss import Loss

class MAE(Loss):

    def fct(self, Y, A):
        return np.mean(np.abs(A - Y))

    def deriv(self, Y, A):
        return np.where(A > Y, 1., -1.)
