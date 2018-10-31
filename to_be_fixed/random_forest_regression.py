# coding: utf-8

# Decision Tree Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

# Feature Scaling

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting SVR to the dataset

rand_forest_reg = RandomForestRegressor(n_estimators=300, random_state=0)
rand_forest_reg.fit(X, y)

# Predicting a new result

y_pred = rand_forest_reg.predict((np.array(6.5).reshape(-1, 1)))
y_pred = rand_forest_reg.predict(sc_X.transform(np.array(6.5).reshape(-1, 1)))
y_pred = rand_forest_reg.inverse_transform(y_pred)

# Visualising the SVR results
# choice of 0.01 instead of 0.1 step because the data is feature scaled

X_grid = np.arange(min(X), max(X), .01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, rand_forest_reg.predict(X_grid), color='blue')
plt.title('Wage v Possition Level (Decision Tree Regression)')
plt.xlabel('Years of Possition Level')
plt.ylabel('Wage')
plt.show()