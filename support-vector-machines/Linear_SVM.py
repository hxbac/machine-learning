import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
# training data
X_train = np.array([[1,0], [0,1], [1,1], [0,2], [4,1], [2,2], [3,2],[1,4],[0,4]])
y_train = np.array([-1, -1, -1, -1, 1, 1, 1, 1, 1])
# SVM
model = SVC(kernel='linear', C = 0.5)
model.fit(X_train, y_train)
# print coefficient
w = model.coef_
b = model.intercept_
print('w = ', w,'b = ',b)
# testing data
z = np.array([[0,0]])
print(model.predict(z))
# plot points
plt.grid(True)
plt.plot(X_train[:4, 0], X_train[:4, 1], 'ro', markersize = 8)
plt.plot(X_train[4:, 0], X_train[4:, 1], 'bs', markersize = 8)
# plot w1x1 + w2x2 + b = 0
x1 = np.linspace(-1,4)
x2 = -(w[0,0]*x1 + b)/w[0,1]
plt.plot(x1,x2, 'b')
plt.show()
