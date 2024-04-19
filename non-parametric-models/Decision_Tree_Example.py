# load libraries
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

columns = ['Outlook', 'Company', 'Sailboat','Sail']
# load dataset
csv = pd.read_csv("example1.csv", header = 0, names = columns)

# split dataset in features and target variable
features = ['Outlook', 'Company', 'Sailboat']
X = csv[features] # features
y = csv.Sail # target variable

# split dataset into training set and test set - 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# create Decision Tree classifer object
model = tree.DecisionTreeClassifier(criterion="entropy")

# train Decision Tree Classifer
model.fit(X_train,y_train)

# predict the response for test dataset
y_pred = model.predict(X_test)

# model accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(X_train)
print(X_test)

# visualizing Decision Trees
tree.plot_tree(model, feature_names = features)
plt.show()