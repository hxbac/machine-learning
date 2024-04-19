import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras import models
from keras import layers
from keras.datasets import mnist
(X_train,y_train),(X_test,y_test) = mnist.load_data()

#normalize the data samples
X_train = X_train.reshape((60000,28*28))
X_train = X_train.astype('float32')/255
X_test = X_test.reshape((10000,28*28))
X_test = X_test.astype('float32')/255

#encode the targers
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#create a network model
model = models.Sequential()

model.add(layers.Dense(256,activation='relu',input_shape = (28*28,)))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#train the network
history = model.fit(X_train, y_train, epochs = 100, verbose = 2, batch_size=128)

#evaluate the trained netwotk
test_loss, test_acc = model.evaluate(X_test, y_test)
print('test_acc:', test_acc)

import numpy as np
print('\n y_test\n')
print(np.argmax(y_test[0:99],axis = 1))

print('\n y_pred\n')
y_pred = np.argmax(model.predict(X_test), axis = 1)
print(np.array(y_pred[0:99]))

