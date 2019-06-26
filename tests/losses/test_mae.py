from losses.mae import MAE
import unittest
import numpy as np

class MAETest(unittest.TestCase):

    def testFunction(self):
        result = MAE.fct(self, np.array([1,2]), np.array([2,1]))
        self.assertEqual(1., result)

    def testDerivation(self):
        result = MAE.deriv(self, np.array([1,2]), np.array([2,1]))
        self.assertEqual(np.array([1., -1.]).tolist(), result.tolist())

