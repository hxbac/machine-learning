# load libraries
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

columns = ['ID', 'Infection', 'Die']
# load dataset
csv = pd.read_csv(r"C:\Users\Bac\Desktop\Hoc may\Codes\Codes\linear-regression\Covid19.csv", header = 0,names=columns)

#split dataset in features and target variable
features = ['ID']
X = csv[features] # features
y = csv.Infection      # farget variable
X = np.array(X)

#normlize samples
n = X.shape[0]
d = 3
x = np.zeros(shape = (d,n))
#y = np.zeros(shape = (1,n))
for i in range(0,n):
    xi = X[i][0]
    x[:, [i]] = np.transpose([[1, xi, xi ** 2]])

#initialize
w = np.zeros(shape = (d,1))
eta = 0.0000001
theta = 0.001
k = 0
#loop
while True:
    k = k + 1
    deltaJ = np.zeros(shape=(d, 1))
    for i in range(n):
        deltaJ = deltaJ + (y[i] - np.matmul(w.transpose(),x[:,[i]]))*x[:,[i]]
    w = w + eta*deltaJ
    if eta*np.linalg.norm(deltaJ) < theta:
        break
print(w)
#plot
plt.plot(X,y,'ro')
t = np.arange(1,25,0.01)
f = w[0,0] + w[1,0]*t + w[2,0]*(t**2)
plt.plot(t,f,'b')
plt.grid(True)
plt.show()
