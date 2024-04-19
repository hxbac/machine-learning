import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import API functions
from keras.datasets import mnist
(X_train,y_train),(X_test,y_test) = mnist.load_data()
# get the image parameters
num_train_samples = X_train.shape[0]
num_test_samples = X_test.shape[0]
height = X_train.shape[1]
width = X_train.shape[2]

# normalize the data samples
X_train = X_train.reshape((num_train_samples, height, width,1))
X_train = X_train.astype('float32') / 255
X_test = X_test.reshape((num_test_samples, height, width,1))
X_test = X_test.astype('float32') / 255

# encode the targers
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# create samples for simple hold-out validation
X_train_val = X_train[:10000]
y_train_val = y_train[:10000]
X_train_par = X_train[10000:]
y_train_par = y_train[10000:]

# create a network model
from keras import models
from keras import layers
num_classes = 10
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), padding = 'same',strides = 1, activation='relu', input_shape=(height,width,1)))
model.add(layers.Conv2D(32, (3, 3), padding = 'same',strides = 1, activation='relu'))
model.add(layers.MaxPooling2D((2, 2),padding = 'same',strides = 2))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

#train the network
history = model.fit(X_train_par, y_train_par, epochs = 25, batch_size = 128, validation_data=(X_train_val, y_train_val))

#evaluate the trained netwotk
test_loss, test_acc = model.evaluate(X_test, y_test)
print('test_acc:', test_acc)

import numpy as np
print(np.argmin(history.history['val_loss']))

#plot training loss and validation loss
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(history.history['loss'],'-b^',label = 'Training loss')
plt.plot(history.history['val_loss'],'-rv',label = 'Validation loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(loc = 'upper right')

#plot training accurancy and validation accurancy
plt.figure(2)
plt.plot(history.history['accuracy'],'-b>',label = 'Training accuracy')
plt.plot(history.history['val_accuracy'],'-r<',label = 'Validation accuracy')
plt.ylabel('Accurancy')
plt.xlabel('Epochs')
plt.legend(loc = 'upper left')
plt.show()