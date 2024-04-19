from sklearn import neighbors, datasets
iris = datasets.load_iris()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=50)

# k = 2, p = 1 (manhattan_distance), p = 2 (euclidean_distance)
model = neighbors.KNeighborsClassifier(n_neighbors = 5, p = 2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Results for 50 test data points:")
print("Predicted labels: ", y_pred)
print("Ground truth    : ", y_test)

from sklearn.metrics import accuracy_score
print("Accuracy of 2NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
