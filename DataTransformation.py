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


# %%
def prepare_data():
    
    data = pd.read_csv('btc.csv')
    data = data[365*2:-1]

    data['Date'] = pd.to_datetime(data['time'])
    data.set_index('Date', inplace=True)
    data['LogPriceUSD'] = np.log(data['PriceUSD'])

    data = data[['LogPriceUSD']]
    data.fillna(0, inplace=True)
    
    return data

data = prepare_data()


# %%
def log_fit(x, a, b, c):
    
    return a * np.log(x + c) + b


def log_trend(X, Y, get_param=False):
    """
    Fits a logarithmic trend to a given dataset.    
    This function estimates the parameters of a logarithmic curve of the form: 
    
        Y = a * log(X + b) + c

    where `X` is a series of integer-indexed dates, and `Y` is the corresponding series of float values.
    
    ----------
    X : array-like
        A series of date indices in integer form.
    Y : array-like
        A series of float values corresponding to the dependent variable.
        
    get_param : bool, optional
        If `get_param` is False, returns an array of fitted trend values.
        If `get_param` is True, returns a tuple `(a, b, c)`, which are the parameters of the logarithmic trend.
    """
    
    initial_guess = [1, 1, 1]
    fitted_param, _ = curve_fit(log_fit, X, Y, p0=initial_guess)
    a, b, c = fitted_param
    
    trend = log_fit(X, a, b, c)
    if get_param: return a, b, c
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
    
    a, b, c = log_trend(X, Y, get_param=True)
    return a, b, c

a, b, c = get_log_params(data, "LogPriceUSD")


# %%
def transform_col(df, name, plot=False):
    
    df = df[[f'{name}']]
    
    ############### Log Transform ###############
    
    Y = pd.Series(df[f'{name}']) 
    X = (Y.index - Y.index[0]).days

    df[f'{name}_log_trend'] = log_trend(X, Y, get_param=False)
    df[f'{name}_residuals'] = df[f'{name}'] - df[f'{name}_log_trend']

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
        axs[0].plot(df.index, df[f'{name}_log_trend'], label='Logarithmic Trend', color='red', linestyle='--')
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

    all_data = all_data[['Difficulty_residuals', 
                         'Transaction Count_residuals', 
                         'Active Addresses Count_residuals',
                         '30 Day Active Supply_residuals',
                         '1 Year Active Supply_residuals',
                         'LogPriceUSD_residuals']]
    return all_data

data = transform_df(data)

