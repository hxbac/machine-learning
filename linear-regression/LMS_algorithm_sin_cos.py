import matplotlib.pyplot as plt
import numpy as np
import math
#define samples
X = np.array([-3,-2,-1,0,1,2,3,4,5])
y = np.array([2.8,0.9,-0.9,-1,0.8,2.7,3.1,1.6,-0.5])
n = X.shape[0]

#normlize samples
d = 3
x = np.zeros(shape = (d,n))
for i in range(0,n):
    x[:, [i]] = np.transpose([[1, np.sin(X[i]), np.cos(X[i])]])

#initialize
w = np.zeros(shape = (d,1))
eta = 0.001
theta = 0.00001
k = 0
#loop
while True:
    k = k + 1
    deltaJ = np.zeros(shape=(d, 1))
    for i in range(n):
        deltaJ = deltaJ + (y[i] - np.dot(w.transpose(),x[:,[i]]))*x[:,[i]]
    w = w + eta*deltaJ
    if eta*np.linalg.norm(deltaJ) < theta:
        break
print(w)

#plot
plt.plot(X,y,'ro')
t = np.arange(-3,6,0.01)
f = w[0, 0] + w[1, 0] * np.sin(t) + w[2, 0] * np.cos(t)
plt.plot(t,f,'r')
plt.grid(True)
plt.show()