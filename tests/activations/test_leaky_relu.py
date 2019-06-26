from activations.leaky_relu import LeakyRelu
import unittest
import numpy as np

class LeakyReluTest(unittest.TestCase):

    """ Testing our LeakyRelu class """

    def testActivation(self):
        """ Testing the formula of our LeakyRelu """
        result = LeakyRelu.activ(self, 12)
        self.assertEqual(12, result)

    def testDerivation(self):
        """ Testing the Derivative of our LeakyRelu """
        np.random.seed(1)
        X = np.random.randn(1,3)
        Y = np.array([[1.0, 0.01, 0.01]])
        result = LeakyRelu.deriv(self, X)
        self.assertEqual(Y.tolist(), result.tolist())

    def testHeuristic(self):
        """ Testing the Heuristic of our LeakyRelu """
        result = LeakyRelu.heuristic(self, 12)
        self.assertEqual(0.408248290463863, result)
