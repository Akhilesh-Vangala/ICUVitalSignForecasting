"""Transformer model for vital sign forecasting."""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerForecaster(nn.Module):
    """Transformer model for vital sign forecasting."""
    
    def __init__(self, input_size: int, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 4,
                 dim_feedforward: int = 512, dropout: float = 0.1,
                 forecast_horizon: int = 6):
        """
        Initialize transformer model.
        
        Args:
            input_size: Number of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            forecast_horizon: Steps ahead to forecast
        """
        super(TransformerForecaster, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, input_size * forecast_horizon)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            
        Returns:
            Forecast tensor (batch_size, forecast_horizon, input_size)
        """
        batch_size, seq_len, _ = x.shape
        
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        
        encoded = self.transformer_encoder(x)
        
        last_hidden = encoded[-1, :, :]
        
        output = self.decoder(last_hidden)
        output = output.view(batch_size, self.forecast_horizon, self.input_size)
        
        return output
