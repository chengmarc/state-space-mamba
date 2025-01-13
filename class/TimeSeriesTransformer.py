# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 21:12:54 2025

@author: Admin
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import torch.nn as nn


# %%
class TimeSeriesTransformer(nn.Module):
    
    def __init__(self, input_dimension, output_length, dropout_rate, num_layers, d_model, nhead):
        
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dimension, d_model),
            nn.Dropout(dropout_rate)  # Dropout after input projection
        )
        
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers, 
            dropout=dropout_rate,  # Dropout in transformer layers
            batch_first=True
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, output_length),
            nn.Dropout(dropout_rate)  # Dropout after the final projection
        )

    def forward(self, source, enforced_target):
        # Project input dimensions to d_model
        source = self.input_proj(source)
        enforced_target = self.input_proj(enforced_target)

        # Pass through the transformer
        transformer_output = self.transformer(source, enforced_target)

        # Project back to output_length
        output = self.output_proj(transformer_output)
        return output[:, -1, :]  # Only return the last time step's forecast


def create_model(data, forecast_horizon, device):
    
    input_dimension = len(data.columns)
    output_length = forecast_horizon
    dropout_rate = 0
    
    num_layers = 5
    d_model = 128
    nhead = 8

    model = TimeSeriesTransformer(input_dimension, output_length, dropout_rate, num_layers, d_model, nhead).to(device)
    print("Time Series Transformer model successfully initialized.")
    return model

