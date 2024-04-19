import numpy as np

x = np.array([[1,1,1],
              [2,2,2],
              [3,3,3],
              [4,4,4],
              [5,5,5],
              [6,6,6],
              [7,7,7],
              [8,8,8],
              [9,9,9]])
y = np.array([1,2,3,4,5,6,7,8,9])
#print(x)

k = np.arange(0,9)
np.random.shuffle(k)
x = x[k,:]
y = y[k]
#k = np.random.shuffle(x)
print(k)
print(x)
print(y)
