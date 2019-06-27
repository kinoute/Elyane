from .activation import Activation
import numpy as np

class Sigmoid(Activation):

    def activ(self, F):
        return 1 / (1 + np.exp(- F))

    def deriv(self, F):
        return F * (1 - F)

    def heuristic(self, F):
        return np.sqrt(1 / F)
