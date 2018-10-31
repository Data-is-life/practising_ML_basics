
# coding: utf-8

# Simple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./Salary_Data.csv')
print(f'{df.head()}')
print('---------------------------------------------------')
print(f'{df.describe()}')
print('---------------------------------------------------')
print(f'{df.info()}')
print('---------------------------------------------------')
print(f'{df.columns}')

X = df['YearsExperience'].values
y = df['Salary'].values
X = X.reshape(X.shape[0], 1)
y = y.reshape(y.shape[0], 1)

print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')

# Splitting data between train and test

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 3, random_state=0)

# Feature scaling

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Fitting simple linear regression to the training set

lin_reg = LinearRegression()
lin_reg = lin_reg.fit(X_train, y_train)

# Predicting the test  set values

y_pred = lin_reg.predict(X_test)

# Plotting the training set results

plt.scatter(X_train, y_train, c='r')
plt.plot(X_train, lin_reg.predict(X_train), c='b')
plt.title('Wage v Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Wage')
plt.show()

# Plotting the test set results

plt.scatter(X_test, y_test, c='r')
plt.plot(X_test, y_pred, c='b')
plt.title('Wage v Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Wage')
plt.show()