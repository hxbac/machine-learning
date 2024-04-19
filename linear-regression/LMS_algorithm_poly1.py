import matplotlib.pyplot as plt
import numpy as np
#define samples
X = np.array([-2,-1,0,1,2])
y = np.array([2,1.5,1,0.5,0])
n = X.shape[0]

#normlize samples
d = 2
x = np.zeros(shape = (d,n))
for i in range(0,n):
    x[:,[i]] = np.transpose([[1, X[i]]])

#initialize
w = np.zeros(shape = (d,1))
eta = 0.001
theta = 0.0001
graph = True
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
    #plot in loop
    if (graph == True)and(k % 20 == 0):
        plt.plot(X, y, 'ro')
        t = np.arange(-5, 10, 0.01)
        f = w[0,0] + w[1,0] * t
        plt.plot(t, f, 'b')
        plt.draw()
        plt.pause(1)
print(w)
#plot
plt.plot(X,y,'ro')
t = np.arange(-5,10,0.01)
f = w[0,0] + w[1,0]*t
plt.plot(t,f,'r')
plt.show()
