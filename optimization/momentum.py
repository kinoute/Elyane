from .optimization import Optimization
import numpy as np

class Momentum(Optimization):
	"""docstring for Momentum"""
	def __init__(self):

		self.Vdw = 0
		self.Vdb = 0

	def for_dw(self, dW):

		self.Vdw = (0.9 * self.Vdw) + (0.1 * dW)

		return self.Vdw

	def for_db(self, db):
		
		self.Vdb = (0.9 * self.Vdb) + (0.1 * db)

		return self.Vdb