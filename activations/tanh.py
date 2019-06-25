from .activation import Activation
import numpy as np

class TanH(Activation):

    def activ(self, F):
        return np.tanh(F)

    def deriv(self, F):
        return 1 - np.square(F)

    def heuristic(self, F):
        return np.sqrt(1 / F)
