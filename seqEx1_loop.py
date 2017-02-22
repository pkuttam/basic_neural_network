
import numpy as np
from linearNN import linearNN
from sigmoidNN import sigmoidNN

net1 = linearNN(2,4)
net2 = sigmoidNN(4)
net3 = linearNN(4,1)

for 1 in range(1,1000):
	inv = np.array([3,1,4])

	net1.forward(inv) 
	net1.backward()
	net2.forward(net1.O)
	net2.backward()
	net3.forward(net2.O)
	net3.backward()



