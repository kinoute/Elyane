from losses.loss import Loss
import unittest

class LossTest(unittest.TestCase):

    def testFunction(self):
        self.assertRaises(NotImplementedError, Loss.fct, self)

    def testDerivation(self):
        self.assertRaises(NotImplementedError, Loss.deriv, self)

