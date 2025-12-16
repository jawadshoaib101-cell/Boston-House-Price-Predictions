import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Splitting data
df = pd.read_csv("Boston (1).csv").sample(frac=1)
train = df[:int(0.6*len(df))]
test = df[int(0.6*len(df)):]


# Scaling
def scaling(dataframe):
    x = dataframe.iloc[:, :-1].values
    y = dataframe.iloc[:, -1].values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    data = np.hstack((x, y.reshape(-1, 1)))
    return data, x, y


train, train_x, train_y = scaling(train)
test, test_x, test_y = scaling(test)

# Using the Linear Regression
linear_regression = LinearRegression()
linear_regression.fit(train_x, train_y)
y_pred_linear_reg = linear_regression.predict(test_x)
comparison_linear_regression = pd.DataFrame({
    "Actual": test_y,
    "Prediction": y_pred_linear_reg
})

# Using Ridge Regression (L2)
ridge_regression = Ridge()
ridge_regression.fit(train_x, train_y)
y_pred_ridge = ridge_regression.predict(test_x)
comparison_ridge = pd.DataFrame({
    "Actual": test_y,
    "Prediction": y_pred_ridge
})

# Using Lasso Regression (L1)
lasso_regression = Lasso()
lasso_regression.fit(train_x, train_y)
y_pred_lasso = lasso_regression.predict(test_x)
# MSE and MAE
ridge_mse = mean_squared_error(test_y, y_pred_ridge)
linear_reg_mse = mean_squared_error(test_y, y_pred_linear_reg)
ridge_mae = mean_absolute_error(test_y, y_pred_ridge)
linear_reg_mae = mean_absolute_error(test_y, y_pred_linear_reg)
lasso_reg_mse = mean_squared_error(test_y, y_pred_lasso)
lasso_reg_mae = mean_absolute_error(test_y, y_pred_lasso)
print(f"The MSE for ridge regression: {ridge_mse}. \nThe MSE for linear regression: {linear_reg_mse}."
      f"\nThe MSE for ridge regression: {ridge_mae}. \nThe MAE for linear regression: {linear_reg_mae}"
      f"\nThe MSE for lasso regression: {lasso_reg_mse}. \nThe MAE for linear regression: {lasso_reg_mae}")

plt.scatter(test_y, y_pred_linear_reg)
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.title('Actual vs Predicted MEDV')
plt.plot([min(test_y), max(test_y)], [min(test_y), max(test_y)], color='red', linewidth=2)  # perfect prediction line
plt.show()
