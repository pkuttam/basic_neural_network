import numpy as np

class linearNN:
	def __init__(self,nI,nO):
		self.nI = nI
		self.nO =nO
		#print(type(nI))
		self.W = np.random.rand(nI,nO)
		self.dW = np.zeros([nI,nO])
		self.B = np.random.rand(nO)
		self.dB = np.zeros(nO)
		self.dI = np.zeros([nI,nO])
		
	def forward(self,I):
		self.O = np.dot(I,self.W)
		self.I = I

	def backward(self):
		self.dB = np.ones(self.nO)
		self.dW = np.dot(np.array([self.I]).T,np.array([np.ones(self.nO)]))
		self.dI = self.W




