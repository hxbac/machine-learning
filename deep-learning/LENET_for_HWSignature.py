#By Hoang Huu Viet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import glob
from PIL import Image
from keras import models
from keras import layers

#read the train images
train_dir = 'hws-train/'
num_classes = 79
X_train = []
y_train = []
for i in range(1,num_classes + 1):
    filenames = train_dir  + str(i) + '-*.png'
    #print(filenames)
    for filename in glob.glob(filenames):
        #print(filename)
        y_train.append(i-1)
        image = Image.open(filename)
        X_train.append(np.array(image))

X_train = np.array(X_train)
y_train = np.array(y_train)

#read the test images
test_dir = 'hws-test/'
X_test = []
y_test = []
for i in range(1,num_classes + 1):
    filenames = test_dir  + str(i) + '-*.png'
    for filename in glob.glob(filenames):
        #print(filename)
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

# display the first 100 images
# import matplotlib.pyplot as plt
# fig = plt.figure()
# cols = 10
# rows = 10
# for i in range(1,rows*cols+1):
#     image = X_train[i-1]
#     fig.add_subplot(rows,cols,i)
#     plt.axis('off')
#     plt.imshow(image,cmap = 'gray')
# plt.show()

#get the image parameters
num_train_samples = X_train.shape[0]
num_test_samples = X_test.shape[0]
height = X_train.shape[1]
width = X_train.shape[2]

#normalize the data samples
X_train = X_train.reshape((num_train_samples, height, width,1))
X_train = X_train.astype('float32') / 255
X_test = X_test.reshape((num_test_samples, height, width,1))
X_test = X_test.astype('float32') / 255

#encode the targers
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#create samples for simple hold-out validation
X_train_val = X_train[:200]
y_train_val = y_train[:200]
X_train_par = X_train[200:]
y_train_par = y_train[200:]
#--------------------------------------------------------------------
#LENET with relu function
model = models.Sequential()
model.add(layers.Conv2D(6, (5, 5), padding='same', activation = 'relu', input_shape = (height,width,1)))
model.add(layers.AveragePooling2D((2, 2),strides = 2))
model.add(layers.Conv2D(16, (5, 5), activation='relu'))
model.add(layers.AveragePooling2D((2, 2),strides = 2))
model.add(layers.Flatten())
model.add(layers.Dense(120, activation='relu'))
model.add(layers.Dense(84, activation='relu'))
model.add(layers.Dense(num_classes,activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

#train the network
history = model.fit(X_train_par, y_train_par, epochs = 40, batch_size = 8, verbose=2, validation_data=(X_train_val, y_train_val))

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
