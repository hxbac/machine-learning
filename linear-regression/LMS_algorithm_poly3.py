import matplotlib.pyplot as plt
import numpy as np
# define samples
X = np.array([-2,-1,0,1,2])
y = np.array([7,5,1,1,11])
n = X.shape[0]
# normlize samples
d = 4
x = np.zeros(shape = (d,n))
for i in range(0,n):
    x[:, [i]] = np.transpose([[1, X[i], X[i] ** 2, X[i] ** 3]])
# initialize
w = np.zeros(shape = (d,1))
eta   = 0.001
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
    # plot in loop
    if (graph == True)and(k % 50 == 0):
        plt.plot(X, y, 'ro')
        t = np.arange(-3, 3, 0.01)
        f = w[0, 0] + w[1, 0] * t + w[2, 0] * (t ** 2) + w[3, 0] * (t ** 3)
        plt.plot(t, f, 'b')
        plt.draw()
        plt.pause(1)
print(w)
#plot
plt.plot(X,y,'ro')
t = np.arange(-3,3,0.01)
f = w[0, 0] + w[1, 0] * t + w[2, 0] * (t ** 2) + w[3, 0] * (t ** 3)
plt.plot(t,f,'r')
plt.grid(True)
plt.show()
