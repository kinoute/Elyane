from layers.layer import Layer
import unittest

class LayerTest(unittest.TestCase):

    def testInit(self):
        self.assertRaises(TypeError, Layer, self, 12, 11, "ok")

    def testForward(self):
        self.assertRaises(NotImplementedError, Layer.forward_pass, self, 12)

    def testBackward(self):
        self.assertRaises(NotImplementedError, Layer.backward_pass, self, 12, 1)
