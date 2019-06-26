from activations.relu import Relu
import unittest
import numpy as np


class ReluTest(unittest.TestCase):

    """ Testing our Relu class """

    def testActivation(self):
        """ Testing the formula of our Relu """
        result = Relu.activ(self, 12)
        self.assertEqual(12, result)

    def testDerivation(self):
        """ Testing the Derivative of our Relu """
        np.random.seed(1)
        X = np.random.randn(1,3)
        Y = np.array([[1., 0., 0.]])
        result = Relu.deriv(self, X)
        self.assertEqual(Y.tolist(), result.tolist())

    def testHeuristic(self):
        """ Testing the Heuristic of our Relu """
        result = Relu.heuristic(self, 12)
        self.assertEqual(0.408248290463863, result)
