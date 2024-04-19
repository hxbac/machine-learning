# Load libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'outcome']
# load dataset
csv = pd.read_csv("diabetes.csv", header = 0,names=col_names)

# split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = csv[feature_cols] # Features
y = csv.outcome # Target variable

# Split dataset into training set and test set - 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

X_train_accuracy = []
X_test_accuracy = []
for c in [0.001,0.01,0.1,1,10]:
    model = SVC(kernel= 'linear', C=c)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    X_train_accuracy.append(accuracy_score(y_train, y_train_pred))
    X_test_accuracy.append( accuracy_score(y_test, y_test_pred))

# plot for determining c
c = [0.001,0.01,0.1,1,10]
import matplotlib.pyplot as plt
plt.subplots(figsize=(10, 5))
plt.semilogx(c, X_train_accuracy, color = 'blue', label="Training Accuracy")
plt.semilogx(c, X_test_accuracy,color='red' , label="Testing Accuracy")

plt.grid(True)
plt.xlabel("Cost Parameter C")
plt.ylabel("Accuracy")
plt.legend()
plt.title('Accuracy versus the Cost Parameter C (log-scale)')
plt.show()

c = 1
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