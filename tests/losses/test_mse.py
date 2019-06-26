from losses.mse import MSE
import unittest
import numpy as np

class MSETest(unittest.TestCase):

    def testFunction(self):
        result = MSE.fct(self, np.array([1,2]), np.array([2,1]))
        self.assertEqual(1., result)

    def testDerivation(self):
        result = MSE.deriv(self, np.array([1,2]), np.array([2,1]))
        self.assertEqual(np.array([1., -1.]).tolist(), result.tolist())

