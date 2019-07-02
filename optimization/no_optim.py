from .optimization import Optimization
import numpy as np

class NoOptim(Optimization):
	"""docstring for NoOptim"""
	def __init__(self):
		pass

	def for_dw(self, dW, t):
		return dW

	def for_db(self, db, t):
		return db
