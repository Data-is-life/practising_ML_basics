import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

df = pd.read_csv('Churn_Modelling.csv')

print(f'{df.head()}')
print('\n-------------------------------------------------------------------\n')
print(f'{df.describe()}')
print('\n-------------------------------------------------------------------\n')
print(f'{df.info()}')
print('\n-------------------------------------------------------------------\n')
print(f'{df.columns}')

X = df.loc[:, ['CreditScore', 'Geography', 'Gender',
               'Age', 'Tenure', 'Balance', 'NumOfProducts',
               'HasCrCard', 'IsActiveMember', 'EstimatedSalary']].values
y = df.loc[:, 'Exited'].values

# Encoding categorical data

labelencoder_X_1 = LabelEncoder()
labelencoder_X_2 = LabelEncoder()
onehotencoder = OneHotEncoder(categorical_features=[1])

X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12)


# Fitting XGBoost to the Training set

classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, verbose=2, n_jobs=-1)

print('Accuracies:\n')
[print(num*100) for num in accuracies]
print('\n---------------------------------------------------\n')
print(f'Accuracies mean: {accuracies.mean()*100}%')
print('\n---------------------------------------------------\n')
print(f'Accuracies standard deviation: {accuracies.std()}')