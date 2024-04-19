import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from keras import models
from keras import layers

# the training data
X_train = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
#the training targets
y_train = np.array([[0],[1],[1],[0]], "float32")
# y_train= np.array([[0,0],[1,1],[1,1],[0,0]],"float32")

model = models.Sequential()
# create the first layer of network
model.add(layers.Dense(4, activation = 'tanh',input_dim = 2))
# create the second layer of network
model.add(layers.Dense(1, activation = 'sigmoid'))
# configures the network for training
model.compile(optimizer = 'sgd',loss = 'binary_crossentropy',metrics = 'accuracy')
# train the network
model.fit(X_train,y_train,epochs = 2000, batch_size = 2)
# generates output predictions for the input samples
print((model.predict(X_train)>0.5).astype("int32"))
# get weights and bias of the first layer
print(model.layers[0].get_weights())
# get weights and bias of the second layer
print(model.layers[1].get_weights())

