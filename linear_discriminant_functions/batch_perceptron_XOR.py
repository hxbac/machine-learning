import numpy as np
#define samples
X = np.array([[0,0],[1,1],[1,0],[0,1]],'float32')
#X = np.array([[0,0],[0,2],[1,2],[0,1],[1,1],[2,0]],'float32')
#normlize samples
n = 4
d = 6
y = np.zeros(shape = (d,n))
for i in range(n):
    y[:, [i]] = np.transpose([[1, X[i][0], X[i][1], X[i][0] ** 2, X[i][0] * X[i][1], X[i][1] ** 2]])

#change sign of class
y[:,2:4] = -y[:,2:4]
#perceptron algorithm
a = np.zeros(shape = (y.shape[0],1))
eta = 1
theta = 0.001
k = 0
while True:
    k = k + 1
    y_sum = np.zeros(shape = (y.shape[0],1))
    for i in range(y.shape[1]):
        if np.matmul(a.transpose(),y[:,[i]]) <= 0:
            y_sum = y_sum + y[:,[i]]

    a = a + eta*y_sum
    if eta*np.linalg.norm(y_sum) < theta:
        break
print('iterations:',k)
print(a)
#plot
import matplotlib.pyplot as plt

plt.plot(X[0:2,0],X[0:2,1],'ro')
plt.plot(X[2:4,0],X[2:4,1],'b*')

x1 = np.linspace(-2,5)
x2 = np.linspace(-2,5)
x1,x2 = np.meshgrid(x1,x2)
f = a[0,0] + a[1,0]*x1 + a[2,0]*x2+a[3,0]*x1**2 + a[4,0]*x1*x2 + a[5,0]*x2**2
plt.contour(x1,x2,f,[0])
plt.grid(True)
plt.show()