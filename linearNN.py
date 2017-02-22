# maintained by Prakash K Uttam

# W = [ [W11, W12, W13, ......], [W21, W22, W23, ......], .........]
# dW = d(outLayer)/d(Weight)
# dI = d(outLayer)/d(inputLayer)
# dB = d(outputLayer)/d(inputLayer)
# I = inputLayer
# O = output
# B = biased

# since all the weight matrix,biased and input are only related to individual output.So,there won't be addition of two differential
# in dW and dI

# I1W11 + I2W21 + I3W31 + ..... = O1
# I1W12 + I2W22 + I3W32 + ...... = O2
# ..................................

# dW = [ [ dO1/dW11, dO2/dW12, dO3/dW13, ........], [dO1/dW21, dO2/dW22, dO3/dW23, ......], ..........]
# dW = [ [ x1, x1, x1 .....], [x2, x2, x2, .....], ..........]]

# dI = [ [ dO1/dI1, dO2/dII, dO3/dI1, ......] , [dO1/dI2, dO2/dI2,dO3/dI3, ......],.....] = W

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




