
import numpy as np
from linearNN import linearNN
from sigmoidNN import sigmoidNN

net1 = linearNN(2,2)
net2 = linearNN(2,1)

x = 10*np.random.rand(1000)
y = 5*x -10

lr = 0.001
epoch = 100
for j in range(1,epoch):
 for i in range(0,len(x)-1):
  #forward pass
  net1.forward(np.array([x[i],1]))
  net2.forward(net1.O)
  #backward pass
  net1.backward()
  net2.backward()
  
  dEy = net2.O - y[i] 
  dYwn2 = net2.dW
  dYbn2 = net2.dB
  dEwn2 = dEy*dYwn2
  dEbn2 = dEy*dYbn2
  
  dYh1 = net2.dI
  dEwn1 = dEy*(np.array([dYh1[0]*net1.dW[:,0],dYh1[1]*net1.dW[:,1]]).T)
  dEbn1 = dEy*(np.array([dYh1[0]*net1.dB[0],dYh1[1]*net1.dB[1]]).T)
  
  net1.W = net1.W-lr*dEwn1
  net1.B = net1.B - lr*dEbn1
  
  net2.W = net2.W-lr*dEwn2
  net2.B = net2.B - lr*dEbn2
  if i%100==0:
   print('Error is ' + str(dEy[0]))
  


x_test = 1000
net1.forward(np.array([x_test,1]))
net2.forward(net1.O)
print('for x_test = ')
print(x_test)
print('y_test =  ')
print(net2.O[0])

