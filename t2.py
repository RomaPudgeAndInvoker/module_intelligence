import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

np.random.seed(42)
X_regression = np.random.rand(1000, 1) * 10
y_regression = 2 * X_regression.squeeze() + np.random.randn(1000) * 2

scaler_regression = MinMaxScaler()
X_regression = scaler_regression.fit_transform(X_regression)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_regression, y_regression, test_size=0.2, random_state=42
)

k_values_reg = range(1, 101, 5)
mse_scores = []
r2_scores = []

for k in k_values_reg:
    knn_regressor = KNeighborsRegressor(n_neighbors=k)
    knn_regressor.fit(X_train_reg, y_train_reg)
    y_pred_reg = knn_regressor.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)
    mse_scores.append(mse)
    r2_scores.append(r2)

best_k_reg_mse = k_values_reg[np.argmin(mse_scores)]
best_mse = np.min(mse_scores)

best_k_reg_r2 = k_values_reg[np.argmax(r2_scores)]
best_r2 = np.max(r2_scores)

print("\nKNN Регресор:")
print(f"Найкраще K (MSE): {best_k_reg_mse}, Найменша MSE: {best_mse:.4f}")
print(f"Найкраще K (R^2): {best_k_reg_r2}, Найвищий R^2: {best_r2:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(k_values_reg, mse_scores, marker='o')
plt.title('MSE регресії KNN для різних K')
plt.xlabel('K')
plt.ylabel('MSE')
plt.xticks(k_values_reg)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(X_test_reg, y_test_reg, label='Фактичні значення', alpha=0.6)
plt.scatter(X_test_reg, knn_regressor.predict(X_test_reg), label='Прогнозовані значення', alpha=0.6)
plt.title(f'Порівняння фактичних та прогнозованих значень (K={best_k_reg_mse})')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()