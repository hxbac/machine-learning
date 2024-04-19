import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from keras import models
from keras import layers
# create the training data
X_train = np.array([[1,1],[1,2],[2,-1],[2,0],
                    [-1,2],[-2,1],[-1,-1],[-2,-2]],"float32")
# create the training targets
y_train = np.array([[0,0],[0,0],[0,1],[0,1],
                    [1,0],[1,0],[1,1],[1,1]],"float32")
model = models.Sequential()
# create a network layer
model.add(layers.Dense(2, activation = 'linear',input_dim = 2))
# configure the network for training
model.compile(optimizer = 'sgd',loss = 'mse')
# train the network
model.fit(X_train, y_train, epochs = 200, batch_size = 8)
# generates output predictions for the input samples
print('\n y_pred\n')
print((model.predict(X_train)>0.5).astype("int32"))

# get weights and bias
print(model.get_weights())

