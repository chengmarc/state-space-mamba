# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:32:34 2025

@author: uzcheng
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# %%
data = pd.read_csv('btc.csv')
data = data[365*2:-1]

data['Date'] = pd.to_datetime(data['time'])
data.set_index('Date', inplace=True)

data['Difficulty'] = np.log(data['DiffLast'])
data['Transaction Count'] = np.log(data['TxCnt'])
data['Active Addresses Count'] = np.log(data['AdrActCnt'])

data['30 Day Active Supply'] = np.log(data['SplyAct30d'])-10
data['1 Year Active Supply'] = np.log(data['SplyAct1yr'])-10
data['CurrentSupply'] = np.log(data['SplyCur'])-10
data['LogPriceUSD'] = np.log(data['PriceUSD'])

data = data[['Difficulty', 'Transaction Count', 'Active Addresses Count', '30 Day Active Supply', '1 Year Active Supply', 'CurrentSupply', 'LogPriceUSD']]
data.fillna(0, inplace=True)


# %%
historic_horizon = 4 * 365  # Use the last 4 years to predict
forecast_horizon = 365  # Predict the next year

from DataLoader import create_dataloader
dataloader = create_dataloader(data, historic_horizon, forecast_horizon, device, debug=False)

from SimpleLSTM import create_model
model = create_model(data, forecast_horizon, device)


# %%
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)

model.train()
epochs = 500
for epoch in range(epochs):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        
        # Squeeze y_batch to remove the extra dimension
        y_batch = y_batch.squeeze(-1)  # Remove the last dimension (365, 1 -> 365)
        
        # Compute the loss
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        
    scheduler.step(loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")


# %%
model.eval()
with torch.no_grad():
    predictions = []
    past = torch.tensor(data.iloc[-historic_horizon:, :].values, dtype=torch.float32).unsqueeze(0).to(device)
    pred = model(past)
    predictions.append(pred.cpu().numpy().flatten())  # Flatten the prediction and move to CPU for further processing
    predictions = np.array(predictions).flatten()

# Create DataFrame for predictions
predicted_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=len(predictions))
predicted_df = pd.DataFrame(predictions, index=predicted_dates, columns=['Predicted PriceUSD'])

# Plot results
plt.figure(figsize=(14, 7))
plt.plot(data.index[-historic_horizon:], data['LogPriceUSD'][-historic_horizon:], label='Log PriceUSD', color='blue')
plt.plot(predicted_df.index, predicted_df['Predicted PriceUSD'], label='Predicted PriceUSD (Next 365 Days)', color='red')
plt.title('Actual vs Predicted PriceUSD for Next 365 Days')
plt.xlabel('Date')
plt.ylabel('PriceUSD')
plt.legend()
plt.show()

