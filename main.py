# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:32:34 2025

@author: uzcheng
"""
import os, sys
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)
sys.path.insert(1, rf'{script_path}\class')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

import warnings
warnings.simplefilter(action='ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# %% Preparation
from DataTransformation import a, b, c # Log parameters used for inverse function later
from DataTransformation import data

historic_horizon = 5 * 365  # Use the last 8 years to predict
forecast_horizon = 30  # Predict the next year
data = data[:-forecast_horizon]

from DataLoader import create_dataloader
train_loader, valid_loader = create_dataloader(data, historic_horizon, forecast_horizon, device, debug=False)

from MambaSSM import create_model
model = create_model(data, forecast_horizon, device)

model_name = model.__class__.__name__
force_teaching = "Transformer" in model_name
model = nn.DataParallel(model, device_ids=list(range(1))) # In case of multiple GPUs


# %% Load Existing Model
if not os.path.exists(rf'{script_path}\model'):
    os.makedirs(rf'{script_path}\model')
    
model_list = [rf'{script_path}\model\{x}' for x in os.listdir(rf'{script_path}\model')]
if model_list:
    model_list.sort(key=lambda x: os.path.getmtime(x))
    model.load_state_dict(torch.load(model_list[-1], weights_only=True)) #load latest model
    print(f'{model_list[-1]} Loaded.')


# %% Training Loop
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

epochs = 1000
for epoch in range(epochs):    
    
    model.train()
    train_loss = 0    
    for inputs, targets in train_loader:        
        optimizer.zero_grad()
        
        if force_teaching: outputs = model(inputs, torch.zeros_like(inputs))
        else: outputs = model(inputs)
            
        outputs = outputs.squeeze(-1)        
        targets = targets.squeeze(-1)
        
        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping (optional, for LSTMs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.8f}")
    
    model.eval()    
    valid_loss = 0
    with torch.no_grad():
        
        for inputs, targets in valid_loader:
            
            outputs = model(inputs)
            outputs = outputs.squeeze(-1)
            targets = targets.squeeze(-1)

            loss = criterion(outputs, targets)
            valid_loss += loss.item()
            
    avg_valid_loss = valid_loss / len(valid_loader)  # Average loss for validation
    print(f"Validation Loss: {avg_valid_loss:.6f}")
    
    scheduler.step(avg_valid_loss)
    
    if epoch % 20 == 0:
        torch.save(model.state_dict(), rf'{script_path}\model\{model_name}-loss-{loss.item():.6f}.pt')
        print(rf'{script_path}\model\{model_name}-loss-{loss.item():.6f}.pt Saved.')


# %% Evaluation
model.eval()

test_period = 2*365
timerange = list(range(1, test_period, 10))
timerange.reverse()

for timeback in timerange:
    with torch.no_grad():
        predictions = []
        past = torch.tensor(data.iloc[-historic_horizon-timeback:-timeback, :].values, dtype=torch.float32).unsqueeze(0).to(device)
        
        if force_teaching: pred = model(past, torch.zeros_like(past))
        else: pred = model(past)
            
        predictions.append(pred.cpu().numpy().flatten())  # Flatten the prediction and move to CPU for further processing
        predictions = np.array(predictions).flatten()

    # Create DataFrame for predictions
    predicted_dates = pd.date_range(start=data.index[-1-timeback] + pd.Timedelta(days=1), periods=len(predictions))
    predicted_price = pd.DataFrame(predictions, index=predicted_dates, columns=['Predicted LogPriceUSD Residual'])

    # Plot results
    plt.figure(figsize=(14, 7))
    plt.plot(data.index[-test_period:], data['LogPriceUSD_residuals'][-test_period:], label='Actual LogPriceUSD Residual', color='blue')
    plt.plot(predicted_price.index, predicted_price['Predicted LogPriceUSD Residual'], label='Predicted LogPriceUSD Residual', color='red')
    plt.title('Actual vs Predicted LogPriceUSD Residual')
    plt.xlabel('Date')
    plt.ylabel('Residual')
    plt.grid(linestyle = '-.')
    plt.legend()
    plt.show()
    
    
# %% Prediction
raw = pd.read_csv(rf'{script_path}/btc.csv', usecols=["time", "PriceUSD"])[-test_period:-1]
raw['Date'] = pd.to_datetime(raw['time'])
raw.set_index('Date', inplace=True)

X = (predicted_price.index - data.index[0]).days
Z = a * np.log(X + c) + b
Y = np.exp(predicted_price.copy().iloc[:, 0] + Z.values)

plt.figure(figsize=(14, 7))
plt.plot(raw.index, raw['PriceUSD'], label='Actual Price (USD)', color='blue')
plt.plot(Y.index, Y, label='Predicted Price (USD)', color='red')
plt.title('Actual vs Predicted Price (USD)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(linestyle = '-.')
plt.legend()
plt.show()

