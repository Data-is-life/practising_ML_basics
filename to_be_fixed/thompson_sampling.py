# coding: utf-8

# Thompson Sampling

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import *

df = pd.read_csv('Ads_CTR_Optimisation.csv')

print(f'{df.head()}')
print('---------------------------------------------------')
print(f'{df.describe()}')
print('---------------------------------------------------')
print(f'{df.info()}')
print('---------------------------------------------------')
print(f'{df.columns}')

# Implementing Thompson Sampling Algorithm

# Visualizing the results

df_reward_sum = []
for num in df.columns:
    df_reward_sum.append(df[num].sum())
df_reward_sum
num_rewards_0
num_rewards_1
total_reward

plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
