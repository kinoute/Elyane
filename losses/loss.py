""" Abstract Class / Interface of our losses functions  """


class Loss:
    """ This class contains the needed functions for our classes """

    def fct(self, labels, preds):
        """ Every loss class sould have a fct method.

        Raises:
            NotImplementedError: Description
        """

        raise NotImplementedError

    def deriv(self, labels, preds):
        """every loss class should have a derivative method.

        Raises:
            NotImplementedError: Description
        """

        raise NotImplementedError
