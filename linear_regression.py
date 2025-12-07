import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

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
y_pred = linear_regression.predict(test_x)
comparison = pd.DataFrame({
    "Actual": test_y,
    "Prediction": y_pred
})
print(comparison.head(30))
plt.scatter(test_y, y_pred)
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.title('Actual vs Predicted MEDV')
plt.plot([min(test_y), max(test_y)], [min(test_y), max(test_y)], color='red', linewidth=2)  # perfect prediction line
plt.show()
