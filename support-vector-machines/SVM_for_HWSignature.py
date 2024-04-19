# By Hoang Huu Viet
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import glob
from PIL import Image

# read the train images
train_dir = 'hws-train/'
num_classes = 79
X_train = []
y_train = []
for i in range(1,num_classes + 1):
    filenames = train_dir  + str(i) + '-*.png'
    for filename in glob.glob(filenames):
        y_train.append(i-1)
        X_image = Image.open(filename)
        X_image = np.array(X_image)
        X_train.append(X_image)

X_train = np.array(X_train)
y_train = np.array(y_train)

# read the test images
test_dir = 'hws-test/'
X_test = []
y_test = []
for i in range(1,num_classes + 1):
    filenames = test_dir  + str(i) + '-*.png'
    for filename in glob.glob(filenames):
        y_test.append(i-1)
        X_image = Image.open(filename)
        X_image = np.array(X_image)
        X_test.append(X_image)

X_test = np.array(X_test)
y_test = np.array(y_test)

# create random rows
r = np.arange(0,len(y_train))
np.random.shuffle(r)
X_train = X_train[r,:]
y_train = y_train[r]

# get the image parameters
num_train_samples = X_train.shape[0]
num_test_samples = X_test.shape[0]
height = X_train.shape[1]
width = X_train.shape[2]

# normalize the data samples
X_train = X_train.reshape(num_train_samples, height*width)
X_train = X_train.astype('float32') / 255
X_test = X_test.reshape(num_test_samples, height*width)
X_test = X_test.astype('float32') / 255

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# determine c
X_train_accuracy = []
X_test_accuracy = []
for c in [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]:
    model = SVC(kernel= 'linear', C=c)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    X_train_accuracy.append(accuracy_score(y_train, y_train_pred))
    X_test_accuracy.append(accuracy_score(y_test, y_test_pred))

# plot for determining c
c = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
import matplotlib.pyplot as plt
plt.subplots(figsize=(10, 5))
plt.semilogx(c, X_train_accuracy,'-gD', color = 'blue', label="Training Accuracy")
plt.semilogx(c, X_test_accuracy,'-gD', color='red' , label="Testing Accuracy")

plt.grid(True)
plt.xlabel("Cost Parameter C",fontsize=14)
plt.ylabel("Accuracy",fontsize=14)
plt.legend()
plt.title('Accuracy versus the Cost Parameter C (log-scale)')
plt.show()

# run for c = 10000
c = 10000
model = SVC(kernel= 'linear', C=c)
model.fit(X_train, y_train)
# predict
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
# accuracy
X_train_accuracy = accuracy_score(y_train, y_train_pred)
X_test_accuracy = accuracy_score(y_test, y_test_pred)
#
print(X_train_accuracy)
print(X_test_accuracy)
#'''