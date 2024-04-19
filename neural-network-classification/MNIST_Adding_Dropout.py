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

#create samples for simple hold-out validation
X_train_val = X_train[:10000]
y_train_val = y_train[:10000]
X_train_par = X_train[10000:]
y_train_par = y_train[10000:]
#--------------------------------------------------------------------
#create the network models
num_epochs = 40
#create a network model without adding dropout
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape =(28*28,)))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history1 = model.fit(X_train_par, y_train_par, epochs = num_epochs, batch_size=128, verbose = 0,validation_data=(X_train_val, y_train_val))
#evaluate the trained netwotk
test_loss, test_acc = model.evaluate(X_test, y_test)
print('test_acc:', test_acc)

#create a network model with adding dropout
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape =(28*28,)))
model.add(layers.Dropout(0.35))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history2 = model.fit(X_train_par, y_train_par, epochs = num_epochs, batch_size=128, verbose = 0,validation_data=(X_train_val, y_train_val))
#evaluate the trained netwotk
test_loss, test_acc = model.evaluate(X_test, y_test)
print('test_acc:', test_acc)

#plot validation loss
import matplotlib.pyplot as plt
plt.plot(history1.history['val_loss'],'-b+',label = 'The model without adding dropout')
plt.plot(history2.history['val_loss'],'-rx',label = 'The model with adding dropout')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(loc = 'upper right')
plt.show()
