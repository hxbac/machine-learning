# Load libraries
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# import API functions
from keras.datasets import mnist
(X_train,y_train),(X_test,y_test) = mnist.load_data()

# get the image parameters
num_train_samples = X_train.shape[0] # 60000
num_test_samples = X_test.shape[0]   # 10000
height = X_train.shape[1] # 28
width = X_train.shape[2]  # 28

# normalize the data samples
X_train = X_train.reshape(num_train_samples, height*width) # 28x28 -> 784x1
X_train = X_train.astype('float32') / 255
X_test = X_test.reshape(num_test_samples, height*width) # 28x28 -> 784x1
X_test = X_test.astype('float32') / 255

# Create Decision Tree classifer object
model = tree.DecisionTreeClassifier()

# Train Decision Tree Classifer
model.fit(X_train,y_train)

#Predict the response for test dataset
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Training Accuracy:",metrics.accuracy_score(y_train, y_train_pred))
print("Test Accuracy:",metrics.accuracy_score(y_test, y_test_pred))