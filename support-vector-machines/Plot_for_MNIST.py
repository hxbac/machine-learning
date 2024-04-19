import warnings
warnings.filterwarnings('ignore')
from keras.datasets import mnist
(X_train,y_train),(X_test,y_test) = mnist.load_data()
print(X_train.shape)
print(X_test.shape)

import matplotlib.pyplot as plt
fig=plt.figure()
cols = 20
rows = 10
for i in range(1,rows*cols+1):
    image = X_train[i-1]
    fig.add_subplot(rows,cols,i)
    plt.axis('off')
    plt.imshow(image,cmap = 'binary')
plt.show()