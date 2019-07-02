''' Abstract Class / Interface of our losses functions '''


class Loss:
    ''' this class contains the needed functions for our classes  '''

    def fct(self):
        ''' every loss class sould have a fct method '''
        raise NotImplementedError

    def deriv(self):
        ''' every loss class should have a derivative method '''
        raise NotImplementedError
