from .loss import Loss
import numpy as np

class CrossEntropy(Loss):

    def fct(self, Y, A):
        return - (Y * np.log(A) + (1-Y) * np.log(1 - A))

    def deriv(self, Y, A):
        return - (np.divide(Y,A) + np.divide(1-Y,1-A))
