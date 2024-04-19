import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
# XOR dataset and targets
X_train = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
y_train = np.array([0, 0, 1, 1])
fignum = 1
# fit the model
for kernel in ('sigmoid', 'poly', 'rbf'):
    model = SVC(kernel=kernel, gamma=4, coef0=0)
    model.fit(X_train, y_train)

    # plot the line, the points, and the nearest vectors to the plane
    fig, ax = plt.subplots()
    plt.figure(fignum, figsize=(1, 1))
    plt.clf()

    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=80, facecolors='None')
    plt.plot(X_train[:2, 0], X_train[:2, 1], 'ro', markersize = 8)
    plt.plot(X_train[2:, 0], X_train[2:, 1], 'bs', markersize = 8)

    plt.axis('tight')
    x_min = y_min = -2
    x_max = y_max = 3

    X, Y = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = model.decision_function(np.c_[X.ravel(), Y.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(X.shape)
    plt.figure(fignum, figsize=(4, 3))
    CS = plt.contourf(X, Y, np.sign(Z), 200, cmap='jet', alpha=.2)
    plt.contour(X, Y, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
    plt.title(kernel, fontsize=15)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    fignum = fignum + 1

plt.show()