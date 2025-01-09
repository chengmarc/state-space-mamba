# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:27:55 2025

@author: Admin
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

from torch import nn


# %%
class SimpleLSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        
        super(SimpleLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.dropout(out[:, -1, :])
        out = self.layer_norm(out)
        out = self.fc(out)
        return out


def create_model(data, forecast_horizon, device):
    
    input_size = len(data.columns)
    hidden_size = 50
    output_size = forecast_horizon
    dropout_rate = 0.05
    
    model = SimpleLSTM(input_size, hidden_size, output_size, dropout_rate).to(device)
    return model

