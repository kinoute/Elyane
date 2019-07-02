from .activation import Activation
import numpy as np

class LeakyRelu(Activation):

    def activ(self, F):
        return np.where(F > 0, F, F * 0.01)

    def deriv(self, F):
        return np.clip(F > 0, 0.01, 1.0)

    def heuristic(self, F):
        return np.sqrt(2 / F)
