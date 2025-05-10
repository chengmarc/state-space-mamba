# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 17:25:24 2025

@author: Admin
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import pandas_ta as ta


from binance.spot import Spot
client = Spot()
print(client.time())

# %%

def get_df(ticker, interval="1d", total_klines=5000):
    
    all_klines = []
    end_time = None  # Start from the latest data

    while len(all_klines) < total_klines:
        try:
            # Fetch K-lines with pagination
            klines = client.klines(ticker, "1d", limit=1000, endTime=end_time)
            if not klines: break

            all_klines.extend(klines)
            end_time = klines[0][0] - 1
            
        except Exception as e:            
            print(f"Error fetching {ticker}: {e}")
            break

    # Convert to DataFrame
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(all_klines[:total_klines], columns=columns)  # Trim excess data
    
    df = df.apply(pd.to_numeric, errors='coerce')
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df = df.sort_values(by="open_time").reset_index(drop=True)

    return df

data = get_df("BTCUSDT")

data['Date'] = pd.to_datetime(data['open_time'])
data.set_index('Date', inplace=True)

lst = ["open", "high", "low", "close", "volume", "num_trades"]
data = data[lst]
for col in lst:
    data[f'{col}'] = np.log(data[f'{col}'])



# %%
def linear_fit(x, a, b):
    
    return a * x + b


def linear_trend(X, Y, get_param=False):
    a_init = (Y[-1] - Y[0]) / (X[-1] - X[0])  # 斜率
    b_init = np.mean(Y)  # 截距
    p0 = [a_init, b_init]

    # 拟合
    popt, pcov = curve_fit(linear_fit, X, Y, p0=p0, maxfev=5000)
    
    # 获取拟合参数
    a, b = popt
    
    trend = linear_fit(X, a, b)
    if get_param: return a, b
    else: return trend


# %%
def get_log_params(df, name):
    """
    Computes the logarithmic trend parameters for a specified column in a DataFrame.

    This function extracts a time series from the DataFrame, converts its index 
    to a numerical format, and fits a logarithmic trend using `log_trend`.

    ----------
    df : pd.DataFrame
        A DataFrame containing the time series data with a DateTime index.
    name : str
        The name of the column for which to compute the logarithmic trend.
        
    Returns a tuple (a, b, c), which are the fitted parameters of the logarithmic trend:        
        a: Intercept of the logarithmic fit.
        b: Coefficient of the logarithmic term.
        c: Shift parameter to ensure a well-defined logarithm.
    """
    df = df[[f'{name}']]
    
    Y = pd.Series(df[f'{name}']) 
    X = (Y.index - Y.index[0]).days
    
    a, b = linear_trend(X, Y, get_param=True)
    return a, b

a, b = get_log_params(data, "close")


# %%
def transform_col(df, name, plot=False):
    
    df = df[[f'{name}']]
    
    ############### Log Transform ###############
    
    Y = pd.Series(df[f'{name}']) 
    X = (Y.index - Y.index[0]).days

    df[f'{name}_linear_trend'] = linear_trend(X, Y, get_param=False)
    df[f'{name}_residuals'] = df[f'{name}'] - df[f'{name}_linear_trend']

    ############### Residual Scaling ###############
    
    rolling_window=30
    volatility = df[f'{name}_residuals'].rolling(rolling_window).std().dropna()

    Y = volatility.values.reshape(-1, 1)
    X = np.arange(len(volatility)).reshape(-1, 1)  # Time as feature

    def get_vol_trend(X, Y):
        
        regressor = LinearRegression()
        regressor.fit(X, Y)
        
        trend = regressor.predict(np.arange(len(df)).reshape(-1, 1))
        return trend

    df[f'{name}_vol_trend'] = get_vol_trend(X, Y)
    df[f'{name}_scaled_residuals'] = df[f'{name}_residuals'] / df[f'{name}_vol_trend']

    ############### Plotting ###############
    
    def plot_trend_residuals(df):
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))

        axs[0].set_title("Original Data and Logarithmic Trend")
        axs[1].set_title("Residuals of Logarithmic Trend")
        axs[2].set_title('Scaled Residuals of Logarithmic Trend')

        axs[0].plot(df.index, df[f'{name}'], label='Original Data', color='blue')
        axs[0].plot(df.index, df[f'{name}_linear_trend'], label='Logarithmic Trend', color='red', linestyle='--')
        axs[1].plot(df.index, df[f'{name}_residuals'], label='Residuals', color='green')
        axs[2].plot(df.index, df[f'{name}_scaled_residuals'], label='Scaled Residuals')

        axs[0].grid(True)
        axs[1].grid(True)
        axs[2].grid(True)

        axs[0].legend()
        axs[1].legend()
        axs[2].legend()

        plt.tight_layout()
        plt.show()

    if plot: plot_trend_residuals(df)
    return df


# %%
def transform_df(data):
    
    all_data = []
    for column in data.columns:
        all_data.append(transform_col(data, str(column), plot=True))
    all_data = pd.concat(all_data, axis=1)
    
    all_data = all_data[['open_residuals', 
                         'high_residuals', 
                         'low_residuals',
                         'close_residuals',
                         'volume_residuals',
                         'num_trades_residuals']]
    all_data = all_data[[col for col in all_data.columns if col != 'close_residuals'] + ['close_residuals']]
    return all_data

data = transform_df(data)


# %%
def calculate_halving(data):
    
    halving_dates = pd.to_datetime(['2009-01-03', '2012-11-28', '2016-07-09', '2020-05-11', '2024-04-19', '2028-03-20'])
    days_since_last = np.zeros(len(data), dtype=int)
    last_seen_date = halving_dates[0]
    for i, date in enumerate(data.index):
        if last_seen_date is not None:
            days_since_last[i] = (date - last_seen_date).days
        if date in halving_dates:
            last_seen_date = date
            days_since_last[i] = 0  # Reset counter when we hit a tracked date

    data['days_since_halving'] = days_since_last
    data['days_since_halving'] = data['days_since_halving'] / data['days_since_halving'].max()
    return data

#data = calculate_halving(data)

