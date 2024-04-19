#=========================================================================
#By Hoang Huu Viet
#references
# 1. https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
# 2. https://datascienceplus.com/keras-regression-based-neural-networks/
#==========================================================================
import warnings
warnings.filterwarnings('ignore')
#===========================================================================
# 1.read dataset
import pandas as pd
df = pd.read_excel('cars.xlsx',sheet_name='cars')

# split dataset into train and test
num_train_samples = 950
X_train, X_test = df.iloc[0:num_train_samples,0:5].values, df.iloc[num_train_samples:,0:5].values
y_train, y_test = df.iloc[0:num_train_samples,5].values, df.iloc[num_train_samples:,5].values
print(X_train.shape[1])
#print(X_train)
#print('\n\n\n')
#print(X_test)
#============================================================================
# 2. Scale dataset
#=============================================================================
# 2.1. Scaled Input Variables
# created scaler
from sklearn.preprocessing import MinMaxScaler
input_scaler = MinMaxScaler()
# fit scaler
input_scaler.fit(X_train)
# transform training dataset
X_train = input_scaler.transform(X_train)
# transform test dataset
X_test = input_scaler.transform(X_test)
#======================================================
# 2.2. Scaled Output Variables
#
# reshape 1d arrays to 2d arrays
y_train = y_train.reshape(len(y_train), 1)
y_test = y_test.reshape(len(y_test), 1)
# created scaler
from sklearn.preprocessing import StandardScaler
output_scaler = StandardScaler()
# fit scaler on training dataset
output_scaler.fit(y_train)
# transform training dataset
y_train = output_scaler.transform(y_train)
# transform test dataset
y_test = output_scaler.transform(y_test)
#=============================================================================
# predict dataset with MLP
# use libraries
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD

# define model
model = Sequential()
model.add(Dense(16, activation='relu', input_dim = X_train.shape[1], kernel_initializer='normal'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# compile model
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9))

# fit model
#history = model.fit(X_train, y_train, epochs=100, verbose=0)
history = model.fit(X_train, y_train, epochs=500, verbose = 2, batch_size = 4, validation_split = 0.1)
print(history.history.keys())

# evaluate the model
train_mse = model.evaluate(X_train, y_train, verbose=0)
test_mse = model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))

#predict
y_pred = model.predict(X_test)
#print(output_scaler.inverse_transform((y_test)))
print(output_scaler.inverse_transform((y_pred)))

# plot loss during training
import matplotlib.pyplot as plt
plt.title('Loss / Mean Squared Error')
plt.plot(history.history['loss'], label='train lost')
plt.plot(history.history['val_loss'], label='validation lost')
plt.legend()
plt.show()