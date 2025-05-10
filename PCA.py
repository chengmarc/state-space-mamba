# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 00:36:53 2025

@author: Admin
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# %%
df = pd.read_csv('https://raw.githubusercontent.com/coinmetrics/data/refs/heads/master/csv/btc.csv')
df = df.drop(columns=['time', 'principal_market_price_usd', 'principal_market_usd'])
df.fillna(0, inplace=True)

# %%
for i in range(df.shape[1]):
    col_x = df['PriceUSD']  # First column
    col_y = df.iloc[:, i]  # Third column

    # Plot the dots
    plt.scatter(col_x, col_y)
    plt.xlabel("PriceUSD")  # Label x-axis with the name of the first column
    plt.ylabel(df.columns[i])  # Label y-axis with the name of the third column
    plt.show()