from .optimization import Optimization
import numpy as np

class Adam(Optimization):
	"""docstring for Adam"""

	def __init__(self, size):
		self.Vdw = 0
		self.Vdb = 0
		self.Sdw = 0
		self.Sdb = 0
		self.beta1 = 0.9
		self.beta2 = 0.999
		self.epsilon = 1e-8

	def for_dw(self, dWeights, t):

		# momentum and rmsprop calculation
		self.Vdw = (self.beta1 * self.Vdw) + ((1 - self.beta1) * dWeights)
		self.Sdw = (self.beta2 * self.Sdw) + ((1 - self.beta2) * np.square(dWeights))

		# bias correction
		self.Vdw = self.Vdw / (1 - np.power(self.beta1, t))
		self.Sdw = self.Sdw / (1 - np.power(self.beta2, t))

		return self.Vdw / (np.sqrt(self.Sdw) + self.epsilon)

	def for_db(self, dBias, t):

		# momentum and rmsprop calculation
		self.Vdb = (self.beta1 * self.Vdb) + ((1 - self.beta1) * dBias)
		self.Sdb = (self.beta2 * self.Sdb) + ((1 - self.beta2) * np.square(dBias))

		# bias correction
		self.Vdb /= (1 - np.power(self.beta1, t))
		self.Sdb /= (1 - np.power(self.beta2, t))

		return self.Vdb / (np.sqrt(self.Sdb) + self.epsilon)
