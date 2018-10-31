import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


def import_csv_data(filename, show_info='y', sep_=',', head_er='y'):
    if sep_ == ',' and head_er == 'y':
        df = pd.read_csv(filename)
    elif sep_ == ',':
        df = pd.read_csv(filename, header=None)
    else:
        df = pd.read_csv(filename, sep=sep_, header=None)
    if show_info == 'y':
        print(f'{df.head()}')
        print('\n-------------------------------------------------------------------\n')
        print(f'{df.describe()}')
        print('\n-------------------------------------------------------------------\n')
        print(f'{df.info()}')
        print('\n-------------------------------------------------------------------\n')
        print(f'{df.columns}')
    return df


def get_x_y(df, x_range=[0, -1], y_range=-1, y_re_shape='n', feat_scale=['y', 'X, y'], encode_cat=['n', [1, 2, 3, 4]], split=['y', 0.2, 9]):

    X = df.iloc[:, x_range[0]:x_range = [1]].values
    y = df.iloc[:, y_range:].values

    # Reshaping y if needed
    if y_re_shape == 'y':
        y = y.reshape(-1, 1)

    # Feature scaling
    if feat_scale[0] == 'y':
        if feat_scale[-1] == 'X, y':
            sc_X = StandardScaler()
            sc_y = StandardScaler()
            X = sc_X.fit_transform(X)
            y = sc_y.fit_transform(y)
        elif feat_scale[-1] == 'X'
            sc_X = StandardScaler()
            X = sc_X.fit_transform(X)
        elif feat_scale[-1] == 'y':
            sc_X = StandardScaler()
            X = sc_X.fit_transform(y)

    # Encoding categorical data
    if encode_cat == 'y':
        for range(num) in encode_cat[-1]:
        labelencoder_X_1 = LabelEncoder()
        labelencoder_X_2 = LabelEncoder()
        onehotencoder = OneHotEncoder(categorical_features=[1])

        X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
        X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

        X = onehotencoder.fit_transform(X).toarray()
        X = X[:, 1:]

    # Splitting data between train and test
    if split[0] == 'y':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split[1], random_state=split[2])

    print(f'X shape: {X.shape}')
    print(f'y shape: {y.shape}')

    return X, y, X_train, X_test, y_train, y_test


def get_standard_prediction(classifier, X_test, y_test):

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    accuracy = (cm[0][0] + cm[1][1]) / (cm.sum())
    print(f'Accuracy = {accuracy*100}%')
    return y_pred


def get_cross_val(classifier, X_train, y_train, c_v=10):
    accuracies = cross_val_score(
        estimator=classifier, X=X_train, y=y_train, cv=c_v, n_jobs=-1)
    [print(num * 100) for num in accuracies]
    print(f'Accuracy means: {accuracies.mean()*100}%')
    print(f'Accuracy standard deviation: {accuracies.std()}')
    return accuracies

# Taking care of the missing data
impu = SimpleImputer()
impu.fit(X[:, 1:3])
X[:, 1:3] = impu.transform(X[:, 1:3])

# Encoding categorical data

label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])

# Encoding the independent variable

hot_encoder = OneHotEncoder(categorical_features=[0])
X = hot_encoder.fit_transform(X).toarray()

# Encoding the dependent variable

label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)