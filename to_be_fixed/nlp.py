# coding: utf-8

# Natural Language Processing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

df = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

print(f'{df.head()}')
print('---------------------------------------------------')
print(f'{df.describe()}')
print('---------------------------------------------------')
print(f'{df.info()}')
print('---------------------------------------------------')
print(f'{df.columns}')

# Cleaning the texts

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
corpus = []
for i in range(len(df)):
    review = word_tokenize(df['Review'][i])
    table = str.maketrans('', '', string.punctuation)
    review = [char.translate(table) for char in review]
    review = [word.lower() for word in review if word.isalpha()]
    ps = PorterStemmer()
    review = [ps.stem(num) for num in review if num not in stop_words]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model

cv = CountVectorizer(max_features=1000)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)

# Fitting Naive Bayes to the Training set

classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
accuracy = (cm[0][0] + cm[1][1]) / (cm.sum())
print(f'Accuracy = {accuracy*100}%')