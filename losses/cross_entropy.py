import numpy as np
from .loss import Loss

class CrossEntropy(Loss):

    def fct(Y, A):
        return - (Y*np.log(A) + (1-Y)*np.log(1 - A))

    def deriv(Y, A):
        return - (np.divide(Y,A) + np.divide(1-Y,1-A))
