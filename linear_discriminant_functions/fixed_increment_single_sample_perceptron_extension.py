import numpy as np
#define samples
X = np.array([[1,1],[1,-1],[4,5],[2,2],[0,2],[2,3]],'float32')
n = 6
d = 6
y = np.zeros(shape = (d,n))
for i in range(n):
    y[:,[i]] = np.transpose([[1, X[i][0], X[i][1], X[i][0] ** 2, X[i][0] * X[i][1], X[i][1] ** 2]])
#change the sign of the second class
y[:,3:6] = -y[:,3:6]
#algorithm
a = np.zeros(shape = (d,1))
k = 0
error = [1,1,1,1,1,1]
while np.sum(error) > 0:
    k = (k + 1) % n
    if np.matmul(a.transpose(),y[:,[k]]) <= 0:
        a = a + y[:,[k]]
        error[k] = 1
    else:
        error[k] = 0
print('a = ',a.transpose())
#plot
import matplotlib.pyplot as plt
plt.plot(X[0:3,0],X[0:3,1],'ro')
plt.plot(X[3:6,0],X[3:6,1],'bo')
x1 = x2 = np.linspace(-2,7)
x1, x2 = np.meshgrid(x1,x2)
f = a[0,0] + a[1,0]*x1 + a[2,0]*x2 + a[3,0]*x1**2 + a[4,0]*x1*x2 + a[5,0]*x2**2
plt.contour(x1, x2,f,[0])
plt.grid(True)
plt.show()