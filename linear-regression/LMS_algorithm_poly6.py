import matplotlib.pyplot as plt
import numpy as np
# define samples
X = np.array([0,1,2,3,4,5,6])
y = np.array([0,4,5,3,2,2,5])
n = X.shape[0]
# normlize samples
d = 7
x = np.zeros(shape = (d,n))
for i in range(0,n):
    x[:, [i]] = np.transpose([[1, X[i], X[i] ** 2, X[i] ** 3,X[i] ** 4,X[i] ** 5,X[i] ** 6]])
# initialize
w = np.zeros(shape = (d,1))
eta   = 0.000000000001
theta = 0.0000000001
graph = False
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
        plt.pause(0.1)
print(w)
#plot
plt.plot(X,y,'ro')
t = np.arange(0,6,0.01)
f = w[0, 0] + w[1, 0] * t + w[2, 0] * (t ** 2) + w[3, 0] * (t ** 3) + w[4, 0] * (t ** 4) + w[5, 0] * (t ** 5) + w[6, 0] * (t ** 6)
plt.plot(t,f,'r')
plt.grid(True)
plt.show()
