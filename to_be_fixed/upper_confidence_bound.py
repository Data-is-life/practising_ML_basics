# coding: utf-8

# Upper Confidence Bound

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import *

df = pd.read_csv('Ads_CTR_Optimisation.csv')

print(f'{df.head()}')
print('---------------------------------------------------')
print(f'{df.describe()}')
print('---------------------------------------------------')
print(f'{df.info()}')
print('---------------------------------------------------')
print(f'{df.columns}')

# Implementing UCB

N = 10000
d = 10
ads_selected = []
num_selections = [0] * d
rewards_sum = [0] * d
total_reward = 0
for num in range(N):
    ad = 0
    max_upper_bound = 0
    for i in range(d):
        if (num_selections[i] > 0):
            avg_reward = rewards_sum[i] / num_selections[i]
            delta_i = sqrt(3/2 * log(num + 1) /
                           num_selections[i])
            upper_bound = avg_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    num_selections[ad] = num_selections[ad] + 1
    reward = df.values[num, ad]
    rewards_sum[ad] = rewards_sum[ad] + reward
    total_reward = total_reward + reward

# Visualizing the results

df_reward_sum = []
for num in df.columns:
    df_reward_sum.append(df[num].sum())

plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()