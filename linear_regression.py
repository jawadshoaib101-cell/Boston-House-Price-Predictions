import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
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

# Using Polynomial Linear Regression
model = LinearRegression()  # Creates an object lin reg class
poly_reg = PolynomialFeatures(degree=2)  # Creates an object poly reg features class
poly_x_train = poly_reg.fit_transform(train_x)
model.fit(poly_x_train, train_y)
poly_x_test = poly_reg.transform(test_x)
y_pred_poly = model.predict(poly_x_test)

# MSE and MAE
ridge_mse = mean_squared_error(test_y, y_pred_ridge)
linear_reg_mse = mean_squared_error(test_y, y_pred_linear_reg)
ridge_mae = mean_absolute_error(test_y, y_pred_ridge)
linear_reg_mae = mean_absolute_error(test_y, y_pred_linear_reg)
lasso_reg_mse = mean_squared_error(test_y, y_pred_lasso)
lasso_reg_mae = mean_absolute_error(test_y, y_pred_lasso)
poly_reg_mse = mean_squared_error(test_y, y_pred_poly)
poly_reg_mae = mean_absolute_error(test_y, y_pred_poly)
print(f"The MSE for ridge regression: {ridge_mse}. \nThe MSE for linear regression: {linear_reg_mse}."
      f"\nThe MSE for ridge regression: {ridge_mae}. \nThe MAE for linear regression: {linear_reg_mae}"
      f"\nThe MSE for lasso regression: {lasso_reg_mse}. \nThe MAE for lasso regression: {lasso_reg_mae}"
      f"\nThe MSE for polynomial regression: {poly_reg_mse}. \nThe MAE for polynomial regression: {poly_reg_mae}")

# Graphing
plt.figure(figsize=(8, 5))

# Histogram of actual values
plt.hist(test_y, bins=20, alpha=0.5, label="Actual", color='blue')

# Histogram of predicted values
plt.hist(y_pred_poly, bins=20, alpha=0.5, label="Predicted", color='red')

plt.xlabel("House Price")
plt.ylabel("Frequency")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.show()
