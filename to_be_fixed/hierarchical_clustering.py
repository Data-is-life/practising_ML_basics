# coding: utf-8

# Hierarchical Clustering

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Mall_Customers.csv')
X = df.iloc[:, [3, 4]].values

print(f'{df.head()}')
print('---------------------------------------------------')
print(f'{df.describe()}')
print('---------------------------------------------------')
print(f'{df.info()}')
print('---------------------------------------------------')
print(f'{df.columns}')

# Using the Dendrogram to find the optimal number of clusters

dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting Hierarchical Clustering to the dataset

hc = AgglomerativeClustering(
    n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# Visualizing the clusters

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1],
            s=100, c='r', label='Cl 1', marker='.')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1],
            s=100, c='b', label='Cl 2', marker='.')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1],
            s=100, c='g', label='Cl 3', marker='.')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1],
            s=100, c='c', label='Cl 4', marker='.')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1],
            s=100, c='m', label='Cl 5', marker='.')
plt.title('Clusters of customers')
plt.xlabel('Annual Income ($k)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
