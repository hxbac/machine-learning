import numpy as np
#define samples
X = np.array([[2,1],[1,2],[1,0],[0,1]],'float32')
#normlize samples
n = 4
d = 3
y = np.zeros(shape = (d,n))
for i in range(n):
    y[:,[i]] = np.transpose([[1,X[i][0],X[i][1]]])
#change sign of the second class
y[:,2:4] = -y[:,2:4]
#initialize
a = np.zeros(shape = (d,1))
k = 0
error = [1,1,1,1]
#loop
while np.sum(error) > 0:
    k = (k + 1) % n
    #if np.matmul(a.transpose(),y[:,[k]]) <= 1:
    if np.dot(a.transpose(), y[:, [k]]) <= 1:
        a = a + y[:,[k]]
        error[k] = 1
    else:
        error[k] = 0
print('a = ',a.transpose())
print('k = ',k)
#plot
import matplotlib.pyplot as plt
plt.plot(X[0:2,0],X[0:2,1],'ro')
plt.plot(X[2:4,0],X[2:4,1],'bo')
x1 = x2 = np.linspace(-2,7)
x1,x2 = np.meshgrid(x1,x2)
f = a[0,0] + a[1,0]*x1 + a[2,0]*x2
plt.contour(x1, x2,f,[0])
plt.grid(True)
plt.show()