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
    
    def __init__(self, input_dimension, hidden_size, output_length, dropout_rate, num_layers, num_shortcuts):
        
        super(SimpleLSTM, self).__init__()        
        self.num_layers = num_layers
        
        # Stacked LSTM layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_dimension if i == 0 else hidden_size, hidden_size, batch_first=True)
            for i in range(num_layers)
        ])
        
        # Residual connections for each LSTM
        self.residual_layers = nn.ModuleList([
            nn.Linear(input_dimension if i == 0 else hidden_size, hidden_size) if (i == 0 or input_dimension != hidden_size) else nn.Identity()
            for i in range(num_shortcuts)
        ])
        
        # Normalization layers for each LSTM
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        
        # Dropout and final fully connected layer
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_length)
        self.activation = nn.ReLU()

    def forward(self, x):
        
        out = x
        for i in range(self.num_layers):
            
            # LSTM layer
            lstm_out, _ = self.lstm_layers[i](out)
            
            # Residual connection
            if i % 4 == 0:
                res = self.residual_layers[i//4](out)
                lstm_out = lstm_out + res
            
            # Normalization and activation
            lstm_out = self.layer_norms[i](lstm_out)
            lstm_out = self.activation(lstm_out)
            
            # Update for next layer
            out = lstm_out
            
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
    

def create_model(data, forecast_horizon, device):
    
    input_dimension = len(data.columns)
    output_length = forecast_horizon
    
    hidden_size = 16
    dropout_rate = 0
    num_layers = 16
    num_shortcuts = 4
    
    model = SimpleLSTM(input_dimension, hidden_size, output_length, dropout_rate, num_layers, num_shortcuts).to(device)
    print("Simple LSTM model successfully initialized.")
    return model

