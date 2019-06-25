''' Abstract Class / Interface of our Activation Classes '''
class Activation(object):

    def activ(self, F):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def deriv(self, F):
        raise NotImplementedError

    def heuristic(self, F):
        raise NotImplementedError
