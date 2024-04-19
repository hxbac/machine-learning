import warnings
warnings.filterwarnings('ignore')
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

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
'''
# determine c
X_train_accuracy = []
X_test_accuracy = []
for c in [0.001,0.01,0.1,1,10]:
    model = SVC(kernel= 'linear', C=c)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    X_train_accuracy.append(accuracy_score(y_train, y_train_pred))
    X_test_accuracy.append(accuracy_score(y_test, y_test_pred))

# plot for determining c
c = [0.001,0.01,0.1,1,10]
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
'''
# run for c = 0.1
c = 0.1
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
