from .loss import Loss
import numpy as np

class MultiClassCrossEntropy(Loss):

    def fct(self, Y, A):
        return - np.sum(Y * np.log(A))

    def deriv(self, Y, A):
        return A - Y
