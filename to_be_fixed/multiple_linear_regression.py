# coding: utf-8

# Multiple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
pd.options.display.float_format = '{:,.2f}'.format

df = pd.read_csv('./50_Startups.csv')
print(f'{df.head()}')
print('---------------------------------------------------')
print(f'{df.describe()}')
print('---------------------------------------------------')
print(f'{df.info()}')
print('---------------------------------------------------')
print(f'{df.columns}')

# Getting X & y

X = df.iloc[:, :-1].values
y = df['Profit'].values

# Encoding categorical data

label_encoder_X = LabelEncoder()
X[:, 3] = label_encoder_X.fit_transform(X[:, 3])

# Encoding the independent variable

hot_encoder = OneHotEncoder(categorical_features=[3])
X = hot_encoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap

X = X[:, 1:]

# Splitting data between train and test

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=0)

# Encoding the dependent variable

label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

# Fitting multiple linear regression to the training set

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predicting the test  set values

y_pred = lin_reg.predict(X_test)

# Building the optimal model using Backward Elimination (BE)

uno = np.ones(shape=(50, 1)).astype(int)
X = np.append(arr=uno, values=X, axis=1)

X_opt = X[:, [0, 1, 4]]

lin_reg_OLS = sm.OLS(endog=y, exog=X_opt).fit()
lin_reg_OLS.summary()