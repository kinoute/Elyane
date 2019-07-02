''' Abstract Class / Interface of our Optimization Functions '''
class Optimization:

	def __init__(self):
		raise NotImplementedError

	def opti_algo(self, dW, db, W, b, learning_rate):
		raise NotImplementedError
	
		