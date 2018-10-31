# coding: utf-8

# Apriori Model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import *

df = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

print(f'{df.head()}')
print('---------------------------------------------------')
print(f'{df.describe()}')
print('---------------------------------------------------')
print(f'{df.info()}')
print('---------------------------------------------------')
print(f'{df.columns}')

# Putting all transactions in a list

transactions = []
for i in range(len(df)):
    transactions.append([str(df.values[i, j]) for j in range(len(df.columns))])


''' Using Apriori function to create rules
transactions = the list of transaction we created
			   We want at least 2 items in any basket to make any correlation.

min_lift = 20% is a good number to start with

min_support =  of item purchased devided by the total transactions.
                We want an item to be purchased 3 times a day to be of relevancy.
                Therefore:
                3X7/7500 = 0.0028
              
min_confidence = 3, just because it seems right'''

rules = apriori(transactions, min_support=0.001,
                min_confidence=0.2, min_lift=4, min_length=2)

# Results

results = list(rules)