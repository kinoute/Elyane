from activations.activation import Activation
import unittest

class ActivationTest(unittest.TestCase):

    def testActivation(self):
        self.assertRaises(NotImplementedError, Activation.activ, self, 12)

    def testDerivation(self):
        self.assertRaises(NotImplementedError, Activation.deriv, self, 12)

    def testHeuristic(self):
        self.assertRaises(NotImplementedError, Activation.heuristic, self, 12)



