import numpy as np
#define samples
X = np.array([-2,-1,0,1,2])
y = np.array([2,1.5,1,0.5,0])
n = X.shape[0]

#normlize samples
d = 2
x = np.zeros(shape = (d,n))
for i in range(0,n):
    x[:,[i]] = np.transpose([[1,X[i]]])

#initialize
w = np.zeros(shape = (d,1))
eta = 0.5
theta = 0.1
k = 0
#loop
while True:
    k = k + 1
    deltaJ = np.zeros(shape=(d, 1))
    for i in range(n):
        w = w + eta * (y[i] - np.dot(w.transpose(), x[:, [i]])) * x[:, [i]]
        deltaJ = deltaJ + eta*(y[i] - np.dot(w.transpose(),x[:,[i]]))*x[:,[i]]
        print(w)
    if eta*np.linalg.norm(deltaJ) < theta:
        break
print(w)
print(k)