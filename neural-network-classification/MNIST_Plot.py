import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.datasets import mnist
(X_train,y_train),(X_test,y_test) = mnist.load_data()
print(X_train.shape)
print(y_train)
print(X_test.shape)
print(y_test)

import matplotlib.pyplot as plt
fig=plt.figure()
cols =10
rows = 10
for i in range(1,rows*cols+1):
    digit = X_test[i-1]
    fig.add_subplot(rows,cols,i)
    plt.axis('off')
    plt.imshow(digit,cmap = plt.cm.binary)
plt.show()
