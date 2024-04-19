#By Hoang Huu Viet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import glob
from PIL import Image
from keras import models
from keras import layers

#read the train images
train_dir = 'ifd-train/'
num_classes = 61
X_train = []
y_train = []
for i in range(1,num_classes + 1):
    filenames = train_dir  + str(i) + '-*.jpg'
    for filename in glob.glob(filenames):
        y_train.append(i-1)
        image = Image.open(filename)
        X_train.append(np.array(image))

X_train = np.array(X_train)
y_train = np.array(y_train)

#read the test images
test_dir = 'ifd-test/'
X_test = []
y_test = []
for i in range(1,num_classes + 1):
    filenames = test_dir  + str(i) + '-*.jpg'
    for filename in glob.glob(filenames):
        y_test.append(i-1)
        image = Image.open(filename)
        X_test.append(np.array(image))

X_test = np.array(X_test)
y_test = np.array(y_test)

#create random rows
r = np.arange(0,len(y_train))
np.random.shuffle(r)
X_train = X_train[r,:]
y_train = y_train[r]

#get the image parameters
num_train_samples = X_train.shape[0]
num_test_samples = X_test.shape[0]
height = X_train.shape[1]
width = X_train.shape[2]

#normalize the data samples
X_train = X_train.reshape((num_train_samples,height*width))
X_train = X_train.astype('float32')/255
X_test = X_test.reshape((num_test_samples,height*width))
X_test = X_test.astype('float32')/255

#encode the targers
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#create a network model
model = models.Sequential()
model.add(layers.Dense(256,activation='relu',input_shape=(height*width,)))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(num_classes,activation='softmax'))
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#train the network
history = model.fit(X_train, y_train, epochs = 100, verbose=2, batch_size= 32)

#evaluate the trained netwotk
test_loss, test_acc = model.evaluate(X_test, y_test)
print('test_acc:', test_acc)
#
print('\n y_test\n')
print(np.argmax(y_test,axis = 1))

print('\n y_pred\n')
y_pred = np.argmax(model.predict(X_test), axis = 1)
print(np.array(y_pred))

