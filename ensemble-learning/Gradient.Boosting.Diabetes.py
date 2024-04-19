# load libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# import train_test_split function
from sklearn.model_selection import train_test_split
# import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# read dataset from file .csv
# test 1 & test 3
col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# test2
#col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# load dataset
# test 1 & test 3
csv = pd.read_csv("diabetes3.csv", header = 0,names = col_names)

# test 2
#csv = pd.read_csv("diabetes2.csv", header = 0,names = col_names)

# split dataset in features and target variable
# test1 & test 3
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

#test 2
#feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age']

X = csv[feature_cols] # Features
y = csv['Outcome'] # Target variable

acc = []
for t in range(1,101):
    # split dataset into # 80% training and 20% test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # create gradient boosting classifer object
    clf = GradientBoostingClassifier(n_estimators = 100, learning_rate=1.0, max_depth = 1, random_state = 0)

    # train gradient boosting Classifer
    model = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = model.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    acc.append(metrics.accuracy_score(y_test, y_pred))

print("\n Average accuracy:", np.sum(acc) / len(acc))
