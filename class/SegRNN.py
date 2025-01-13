# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 03:23:38 2025

@author: Admin
"""
import torch
import torch.nn as nn
import torch.optim as optim


# %%
class SegRNN(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, seg_length):
        """
        Initialize SegRNN model.

        Args:
            input_dim (int): Number of features in the input.
            hidden_dim (int): Number of hidden units in the RNN.
            output_dim (int): Number of features in the output.
            num_layers (int): Number of RNN layers.
            seg_length (int): Length of each segment for segmentation.
        """
        super(SegRNN, self).__init__()
        
        self.seg_length = seg_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # RNN layer (you can use LSTM or GRU instead of vanilla RNN)
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer for output
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        batch_size, seq_length, _ = x.size()
        
        # Segment the input sequence
        num_segments = seq_length // self.seg_length
        x = x[:, :num_segments * self.seg_length, :]
        x = x.view(batch_size * num_segments, self.seg_length, -1)
        
        # Pass through RNN
        _, (hidden, _) = self.rnn(x)
        
        # Use the last hidden state of each segment
        hidden = hidden[-1]  # Shape: (batch_size * num_segments, hidden_dim)
        
        # Reshape and aggregate hidden states
        hidden = hidden.view(batch_size, num_segments, self.hidden_dim)
        hidden = torch.mean(hidden, dim=1)  # Aggregate across segments
        
        # Final output
        out = self.fc(hidden)
        return out


def create_model(data, forecast_horizon, device):
# Define model hyperparameters
    input_dim = len(data.columns)  # Number of features in input
    hidden_dim = 64  # Hidden state dimension
    output_dim = forecast_horizon  # Number of steps to forecast (output sequence length y)
    num_layers = 2  # Number of RNN layers
    seg_length = 5  # Length of each segment

    # Instantiate the model
    model = SegRNN(input_dim, hidden_dim, output_dim, num_layers, seg_length)
    return model

