from .activation import Activation
import numpy as np

class Softmax(Activation):

    def activ(self, F):
        #return np.exp(F)/np.sum(np.exp(F), axis=0)
        maxVal = np.max(F, axis = 0, keepdims = True)  # To normalize the values for numerical stability
        return np.exp(F - maxVal) / np.sum(np.exp(F - maxVal), axis = 0, keepdims = True)

    def deriv(self, F):
        return F * (1 - F)
        #deriv = F.reshape(-1, 1)
        #return np.diagflat(deriv) - np.dot(F, deriv.T)

    def heuristic(self, F):
        return np.sqrt(1 / F)
