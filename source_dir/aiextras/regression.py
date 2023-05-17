# regression with sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

data = pd.read_csv('data.csv')
model = LinearRegression()


X = np.array(data['x']).reshape(-1, 1)
y = np.array(data['y'])

model.fit(X, y)

print('Coefficients: \n', model.coef_)
print('Intercept: \n', model.intercept_)
y_pred = model.predict(X)

plt.scatter(data['x'], y, color='black')
plt.plot(data['x'], y_pred, color='blue', linewidth=3)

plt.show()
