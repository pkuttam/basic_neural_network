import numpy as np

class sigmoidNN:
	def __init__(self,nI):
		self.nI=nI
		print(type(nI))
		self.W = np.random.rand(nI)
		self.dW = np.zeros(nI)
		
	def forward(self,I):
		self.O = 1/(1+np.exp(-1*I))
		self.I = I

	def backward(self):
		self.dW = self.O*(1-self.O)




