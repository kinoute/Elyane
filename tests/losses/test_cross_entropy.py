from losses.cross_entropy import CrossEntropy
import unittest
import numpy as np

class CrossEntropyTest(unittest.TestCase):

    def testFunction(self):
        result = CrossEntropy.fct(self, np.array([1., 0., 1.]), np.array([0.75, 0.2, 0.4]))
        self.assertEqual(np.array([0.2876820724517809, 0.2231435513142097, 0.916290731874155]).tolist(), result.tolist())

    def testDerivation(self):
        result = CrossEntropy.deriv(self, np.array([1., 0., 1.]), np.array([0.75, 0.2, 0.4]))
        self.assertEqual(np.array([-1.3333333333333333, 1.25, -2.5]).tolist(), result.tolist())

