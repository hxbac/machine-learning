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

#validate a network model using K-fold validation
import numpy as np
k = 4
num_epochs = 40
num_val_samples = len(X_train) // k
all_loss_histories = []
for i in range(k):
    print('processing fold #', i)

    #create validation samples
    X_train_val = X_train[i * num_val_samples: (i + 1) * num_val_samples]
    y_train_val = y_train[i * num_val_samples: (i + 1) * num_val_samples]
    #create traing samples
    X_train_par = np.concatenate([X_train[:i * num_val_samples],X_train[(i + 1) * num_val_samples:]],axis=0)
    y_train_par = np.concatenate([y_train[:i * num_val_samples],y_train[(i + 1) * num_val_samples:]],axis=0)

    #create a network model
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    #train the network
    history = model.fit(X_train_par, y_train_par, epochs = num_epochs, batch_size=128,verbose=0,validation_data=(X_train_val, y_train_val))
    #save the validation loss
    all_loss_histories.append(history.history['val_loss'])

#average the validation loss for epoches
avg_loss_history = np.mean(all_loss_histories,axis = 0)

#evaluate the trained netwotk
test_loss, test_acc = model.evaluate(X_test, y_test)
print('test_acc:', test_acc)

#plot validation loss
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(avg_loss_history,'r',label = 'Validation loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(loc = 'upper left')
plt.show()
