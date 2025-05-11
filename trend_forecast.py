# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 03:54:41 2025

@author: Admin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter
from scipy.optimize import curve_fit


# %%
df = pd.read_csv('https://raw.githubusercontent.com/coinmetrics/data/refs/heads/master/csv/btc.csv')
df = df[365*2:-1]

df['Date'] = pd.to_datetime(df['time'])
df.set_index('Date', inplace=True)
df['log_price'] = np.log(df['PriceUSD'])
df = df[['log_price']]


# %%    
def log_trend_func(t, a, b, c):
    return a * np.log(t + c) + b

df['time_int'] = np.arange(len(df))
x, y = df['time_int'].values, df['log_price'].values
initial_guess = [1.0, y.mean(), 1.0]

# Fit trend
params, _ = curve_fit(log_trend_func, x, y, p0=initial_guess, maxfev=10000)
a, b, c = params
df['log_trend'] = log_trend_func(x, a, b, c)

# Forecast trend
n_forecast = 500
future_time = np.arange(len(df), len(df) + n_forecast)
future_log_trend = log_trend_func(future_time, a, b, c)

# Forecast bands
residuals = df['log_price'] - df['log_trend']
fixed_residuals = [-1, 0, 1, 2, 3]
bands = {i: df['log_trend'] + i for i in fixed_residuals}
future_bands = {i: future_log_trend + i for i in fixed_residuals}
future_dates = pd.date_range(start=df.index[-1], periods=n_forecast + 1, freq='D')[1:]


# %%
plt.figure(figsize=(12, 8))
plt.plot(df.index, df['log_price'], label='Log Price', color='black')

for i in fixed_residuals:
    label = f'{i} Residual Band'
    linestyle = '-'
    plt.plot(df.index, bands[i], linestyle=linestyle, label=label)

for i in fixed_residuals:
    linestyle = '--'
    plt.plot(future_dates, future_bands[i], linestyle=linestyle, color='gray')

# Customize grid and ticks
plt.grid(True, linestyle='--')
plt.xticks(rotation=45)
plt.yticks(np.arange(-1, 15, 1), np.exp(np.arange(-1, 15, 1)).astype(int))

ax = plt.gca()
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.yaxis.set_minor_locator(MultipleLocator(0.25))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.grid(True, which='minor', linestyle='--', linewidth=0.5, color='gray')
ax.grid(True, which='major', linestyle='-', linewidth=0.5, color='black')

plt.title("Log Trend with Residual Bands")
plt.xlabel("Date")
plt.ylabel("Log Price")
plt.legend()
plt.show()

