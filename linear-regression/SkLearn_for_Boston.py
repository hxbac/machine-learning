import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

boston_data = load_boston()
X = boston_data.data[:, 5, np.newaxis]   # trường thứ 5 là RM (số phòng)
y = boston_data.target[:, np.newaxis]    # taget là giá nhà

# Visualize data
plt.plot(X, y, 'bo')
plt.xlabel('RM')
plt.ylabel('PRICE')
plt.show()

# trước hết cần scale data để tránh trường hợp số lớn làm tràn phép tính
scalerx = StandardScaler()
scalery = StandardScaler()
X_scaled = scalerx.fit_transform(X)
y_scaled = scalery.fit_transform(y)

# chia dữ liệu thành tập train và tập test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled)

# lấy model LinearRegression từ thư viên SKlearn
model = LinearRegression()
# thực hiện train model trên tập train
model.fit(X_train, y_train)

# thực hiện train model trên tập test
y_hat = model.predict(X_test)

# tính sai số lỗi MSE của dự đoán so với thực tế trên tập test
mse = mean_squared_error(y_hat, y_test)
print(mse)

# hiển thị đồ thị dữ liệu test (xanh) và đường dự đoán đã học được (đỏ)
plt.scatter(scalerx.inverse_transform(X_test), scalery.inverse_transform(y_test))
plt.plot(scalerx.inverse_transform(X_test), scalery.inverse_transform(y_hat), 'r')
plt.xlabel('House feature: RM')
plt.ylabel('Price')
plt.show()

# in weight w0, w1 mà model của sklearn đã học w1, w0
print(model.coef_, model.intercept_)

