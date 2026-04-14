from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt


def run_experiment(use_scaler=False, degree=1, ax=None):
   
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

    ###多项式

    if degree > 1:
        poly = PolynomialFeatures(degree=degree)

        X_train = poly.fit_transform(X_train)
        X_test = poly.transform(X_test)

    ###归一化

    if use_scaler:
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    ###引用模型+训练模型

    model = LinearRegression()
    model.fit(X_train,y_train)

    ###模型预测

    y_pred = model.predict(X_test)

    ###模型评估

    mse = mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)

    ###交叉验证

    cv_scores = cross_val_score(model, X_train, y_train, cv = 5)

    cv_mean = cv_scores.mean()

    ###可视化选择

    if ax is not None:
        ax.scatter(y_test, y_pred, alpha=0.3, s=5)

        Max = max(y_pred.max(), y_test.max())
        Min = min(y_pred.min(), y_test.min())

        ax.plot([Min, Max], [Min, Max], linewidth=1.5)

        ax.set_title(f"Scaler={use_scaler}, Degree={degree}")
        ax.set_xlabel('True Price')
        ax.set_ylabel('Predicted Price')

        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.grid(alpha=0.2)

    return mse, r2, cv_mean


###交叉验证（四组实验）

configs = [
    (False, 1),  # 不归一化 + 简单模型
    (True, 1),   # 归一化 + 简单模型
    (False, 2),  # 不归一化 + 复杂模型
    (True, 2)    # 归一化 + 复杂模型
]

###2*2子图

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()


results = []

for i, (use_scaler, degree) in enumerate(configs):
    mse, r2, cv = run_experiment(
        use_scaler=use_scaler,
        degree=degree,
        ax=axes[i]
    )

    results.append((use_scaler, degree, mse, r2, cv))

    print(f"Scaler={use_scaler}, Degree={degree}")
    print(f"MSE={mse:.4f}, R2={r2:.4f}, CV={cv:.4f}")
    print("-" * 40)

plt.tight_layout()
plt.show()