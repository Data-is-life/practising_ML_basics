# coding: utf-8

# Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('./Position_Salaries.csv')
print(f'{df.head()}')
print('---------------------------------------------------')
print(f'{df.describe()}')
print('---------------------------------------------------')
print(f'{df.info()}')
print('---------------------------------------------------')
print(f'{df.columns}')

# Getting X & y

X = df.iloc[:, 1:2].values
y = df['Salary'].values

# Fitting linear regression to the data set

lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting polynomial regression to the data set

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Predicting values

y_lin_pred = lin_reg.predict(X)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid)), 1)

y_pol_pred = lin_reg_2.predict(poly_reg.fit_transform(X_grid))

# Plotting the linear regression results

plt.scatter(X, y, c='r')
plt.plot(X, y_lin_pred, c='b')
plt.title('Wage v Possition Level (Linear only model)')
plt.xlabel('Years of Possition Level')
plt.ylabel('Wage')
plt.show()

# Plotting the linear regression results

plt.scatter(X, y, c='r')
plt.plot(X_grid, y_pol_pred, c='b')
plt.title('Wage v Possition Level (Polynomial model)')
plt.xlabel('Years of Possition Level')
plt.ylabel('Wage')
plt.show()

# Predicting a new result with Linear Regression

lin_reg.predict(np.array(6.5).reshape(-1, 1))

# Predicting a new result with Polynomial Regression

lin_reg_2.predict(poly_reg.fit_transform(np.array(6.5).reshape(-1, 1)))
