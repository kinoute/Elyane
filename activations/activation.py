''' Abstract Class / Interface of our Activation Functions '''
class Activation:

    def activ(self, F):
        raise NotImplementedError

    def deriv(self, F):
        raise NotImplementedError

    def heuristic(self, F):
        raise NotImplementedError
