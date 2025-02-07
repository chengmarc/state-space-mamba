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
from torch.utils.data import DataLoader, TensorDataset, random_split


# %%
def create_dataloader(data, historic_horizon, forecast_horizon, device, debug=False):

    X, y = [], []
    for i in range(len(data) - historic_horizon - forecast_horizon + 1):
        inputs, targets = data[:], data.iloc[:, -1] # all features for price
        X.append(inputs[i:(i + historic_horizon)].values)  # All columns as input features
        y.append(targets[(i + historic_horizon):(i + historic_horizon + forecast_horizon)].values)  # Last column as target

    X, y = np.array(X), np.expand_dims(np.array(y), -1)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    
    print(f"Inputs shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    
    train_size = int(len(dataset) * 0.9)
    valid_size = len(dataset) - train_size

    train_dataset, valid_datase = random_split(dataset, [train_size, valid_size])    
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_datase, batch_size=batch_size, shuffle=False)
    
    if debug: return (X, y)
    else: return train_loader, valid_loader

