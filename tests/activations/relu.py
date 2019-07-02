from .activation import Activation
import numpy as np

class Relu(Activation):

    def activ(self, F):
        return np.maximum(0, F)

    def deriv(self, F):
        return (F > 0).astype(float)

    def heuristic(self, F):
        return np.sqrt(2 / F)
