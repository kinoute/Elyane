from activations.tanh import TanH
import unittest

class TanHTest(unittest.TestCase):

    """ Testing our TanH class """

    def testActivation(self):
        """ Testing the formula of our TanH """
        result = TanH.activ(self, 12)
        self.assertEqual(0.9999999999244973, result)

    def testDerivation(self):
        """ Testing the Derivative of our TanH """
        result = TanH.deriv(self, 12)
        self.assertEqual(-143, result)


    def testHeuristic(self):
        """ Testing the Heuristic of our TanH """
        result = TanH.heuristic(self, 12)
        self.assertEqual(0.28867513459481287, result)
