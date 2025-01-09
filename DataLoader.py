# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:22:54 2025

@author: Admin
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


# %%
def create_dataloader(data, historic_horizon, forecast_horizon, device, debug=False):

    X, y = [], []
    for i in range(len(data) - historic_horizon - forecast_horizon + 1):
        inputs, targets = data[:], data[['LogPriceUSD']]
        X.append(inputs[i:(i + historic_horizon)].values)  # All columns as input features
        y.append(targets[(i + historic_horizon):(i + historic_horizon + forecast_horizon)].values)  # Last column as target

    X, y = np.array(X), np.array(y)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    batch_size = 32
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    if debug: return (X, y)
    else: return dataloader

