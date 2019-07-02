from activations.sigmoid import Sigmoid
import unittest

class SigmoidTest(unittest.TestCase):

    """ Testing our Sigmoid class """

    def testActivation(self):
        """ Testing the formula of our Sigmoid """
        result = Sigmoid.activ(self, 12)
        self.assertEqual(0.9999938558253978, result)

    def testDerivation(self):
        """ Testing the Derivative of our Sigmoid """
        result = Sigmoid.deriv(self, 12)
        self.assertEqual(-132, result)


    def testHeuristic(self):
        """ Testing the Heuristic of our Sigmoid """
        result = Sigmoid.heuristic(self, 12)
        self.assertEqual(0.28867513459481287, result)
