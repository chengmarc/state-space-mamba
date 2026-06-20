# -*- coding: utf-8 -*-
"""ARCHIVED exploratory script — not part of the pipeline.

Quick scatter plots of every CoinMetrics column against PriceUSD, used during
early feature exploration. Superseded by pipeline/step_2_feature_engineering.py.
Kept for reference; run standalone.

Created on Tue Mar 11 00:36:53 2025
@author: Admin
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import pandas as pd
import matplotlib.pyplot as plt


# %%
df = pd.read_csv('https://raw.githubusercontent.com/coinmetrics/data/refs/heads/master/csv/btc.csv')
df = df.drop(columns=['time', 'principal_market_price_usd', 'principal_market_usd'])
df.fillna(0, inplace=True)


for i in range(df.shape[1]):
    col_x = df['PriceUSD']  # First column
    col_y = df.iloc[:, i]  # Third column

    # Plot the dots
    plt.scatter(col_x, col_y)
    plt.xlabel("PriceUSD")  # Label x-axis with the name of the first column
    plt.ylabel(df.columns[i])  # Label y-axis with the name of the third column
    plt.show()

