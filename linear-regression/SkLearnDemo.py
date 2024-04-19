import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Bộ dữ liệu thứ 1 random
np.random.seed(2)
X = np.random.rand(100, 1)
y = -3 * X + 5 + 0.1*np.random.randn(100, 1) # thêm nhiễu vào
# Visualize data
plt.plot(X, y, 'bo')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
#Split X into (X_train, y_train) and (X_test,y_test)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# sử dụng model LinearRegresion từ thư viện Sklearn
model = LinearRegression()
# thực hiện train dữ liệu trên tập train
model.fit(X_train, y_train)

# thực hiện dự đoán trên tập test
y_hat = model.predict(X_test)
# hiển thị MSE của dự đoán so với label
print(mean_squared_error(y_test, y_hat))

print(model.coef_, model.intercept_)

