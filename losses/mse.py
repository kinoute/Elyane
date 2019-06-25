import numpy as np
from loss import Loss

class MSE(Loss):

    def fct(Y, A):
        return np.mean(np.power(Y-A, 2))

    def deriv(Y, A):
        return 2*(A-Y)/Y.size
