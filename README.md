#### DEVELOPED BY : PRAVEEN S
#### REG NO : 212222240078
#### DATE : 


# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

data = pd.read_csv('/content/OnionTimeSeries - Sheet1 (1).csv')

data['Date'] = pd.to_datetime(data['Date'])

data['Min'] = data['Min'].fillna(data['Min'].mean())

X = np.array(data.index).reshape(-1, 1)
y = data['Min']

linear_regressor = LinearRegression()
linear_regressor.fit(X, y)
y_pred_linear = linear_regressor.predict(X)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y)
y_pred_poly = poly_regressor.predict(X_poly)

plt.figure(figsize=(35, 5))
plt.subplot(1,3,1)
plt.plot(data['Date'], data['Min'], label='Price')
plt.xlabel('Date')
plt.ylabel('price')
plt.title('Cost Of Onion')
plt.grid(True)

plt.figure(figsize=(35, 5))
plt.subplot(1,3,2)
plt.plot(data['Date'], y, label='Price')
plt.plot(data['Date'], y_pred_linear, color='red',linestyle='--', label='Linear Trend')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Linear Trend Estimation for Onion Price')
plt.legend()
plt.grid(True)

plt.figure(figsize=(35, 5))
plt.subplot(1,3,3)
plt.plot(data['Date'], y, label='Actual Price')
plt.plot(data['Date'], y_pred_poly, color='green',linestyle='--', label='Polynomial Trend (Degree 2)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Polynomial Trend Estimation for Onion Price')
plt.legend()
plt.grid(True)
plt.show()
```
### OUTPUT
#### A - LINEAR TREND ESTIMATION
![image](https://github.com/user-attachments/assets/803eae35-24fc-4033-8bad-30f0e3313aa5)


#### B- POLYNOMIAL TREND ESTIMATION
![image](https://github.com/user-attachments/assets/472f425e-ef71-427a-8af9-9e64f03e7202)


### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
