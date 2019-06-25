from activations.sigmoid import Sigmoid
import unittest

class SigmoidTest(TestCase):

    """ Testing our Sigmoid class """

    def testActivation():
        """ Testing the formula of our Sigmoid """
        result = Sigmoid.activ(12)
        self.assertEquals(12, result)
