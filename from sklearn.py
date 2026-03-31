from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

###导入数据

data = fetch_california_housing()
X = data.data
y = data.target

mask = y < 5.0
X = X[mask]
y = y[mask]

###切割数据

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.2, random_state = 42
)

###打印数据的shape

print("The train datas:",X_train.shape)
print("The train datas:",X_test.shape)

###引用模型+训练模型

model = LinearRegression()
model.fit(X_train,y_train)

###模型预测

y_pred = model.predict(X_test)

###模型评估

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print(f"mse:{mse:.4f}")
print(f"r2:{r2:.4f}")

###可视化

plt.figure(figsize = (8,6))#创建画布
plt.scatter(y_test, y_pred, alpha = 0.3, color = 'steelblue',s = 5)#创建散点图，透明度，颜色，点的大小

Max = max(y_pred.max(),y_test.max())#找出最大值最小值
Min = min(y_pred.min(),y_test.min())

plt.plot([Min,Max],[Min,Max],
         color = 'red',linewidth = '1.5')#预测线

plt.xlabel('true house price')
plt.ylabel('predict house price')
plt.grid(alpha=0.2) 
plt.xlim(0, 5)
plt.ylim(0, 5)

plt.tight_layout()

plt.show()