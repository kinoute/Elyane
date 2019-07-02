from .optimization import Optimization
import numpy as np

class RMSprop(Optimization):
	"""docstring for Momentum"""

	def __init__(self):
		self.Sdw = 0
		self.Sdb = 0

	def for_dw(self, dWeights, t):

		self.Sdw = (0.999 * self.Sdw) + (0.001 * np.power(dWeights, 2))

		return	dWeights / (np.sqrt(self.Sdw) + 10**-8)

	def for_db(self, dBias, t):

		self.Sdb = (0.999 * self.Sdb) + (0.001 * np.power(dBias, 2))

		return dBias / (np.sqrt(self.Sdb) + 10**-8)
