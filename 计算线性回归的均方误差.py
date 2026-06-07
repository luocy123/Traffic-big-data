import numpy as np
from sklearn.linear_model import LinearRegression
#随机生成训练数据
X_train = np.random.randint(0,20,size=[5000,2])
y_train = np.random.randint(0,20,5000)
#线性拟合
linreg = LinearRegression()
linreg.fit(X_train,y_train)
#计算估计值
y_estimate = linreg.predict(X_train)
#计算估计误差
error_sum = 0
for i in range(5000):
    error_sum += (y_train[i] - y_estimate[i])**2
MSE = error_sum/5000
print("MSE",MSE)