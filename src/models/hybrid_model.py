import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from .lstm_model import LSTMForecaster, TemporalAttention
from .arima_model import ARIMAForecaster
from .tcn_model import TCNForecaster


class AdaptiveFusion(nn.Module):
    def __init__(self, hidden_size: int, num_components: int = 3):
        super(AdaptiveFusion, self).__init__()
        self.num_components = num_components
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_size * num_components, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_components),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, component_outputs: list, hidden_states: list):
        combined_hidden = torch.cat(hidden_states, dim=-1)
        weights = self.fusion_network(combined_hidden)
        
        weighted_output = torch.zeros_like(component_outputs[0])
        for i, output in enumerate(component_outputs):
            weight = weights[:, i].unsqueeze(-1).unsqueeze(-1)
            weighted_output += weight * output
        
        return weighted_output, weights


class HybridLSTMARIMA(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_layers: int = 3, dropout: float = 0.3,
                 forecast_horizon: int = 6, lstm_weight: float = 0.7,
                 use_attention: bool = True, use_tcn: bool = True,
                 use_ensemble: bool = True):
        super(HybridLSTMARIMA, self).__init__()
        
        self.lstm_model = LSTMForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            forecast_horizon=forecast_horizon,
            use_attention=use_attention,
            use_residual=True,
            bidirectional=False
        )
        
        if use_tcn:
            self.tcn_model = TCNForecaster(
                input_size=input_size,
                num_channels=[64, 128, hidden_size],
                kernel_size=3,
                dropout=dropout,
                forecast_horizon=forecast_horizon
            )
        else:
            self.tcn_model = None
        
        self.arima_model = ARIMAForecaster(order=(2, 1, 2))
        
        self.use_attention = use_attention
        if use_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_size)
        
        self.use_ensemble = use_ensemble
        if use_ensemble:
            self.fusion = AdaptiveFusion(hidden_size, num_components=2 if not use_tcn else 3)
        
        self.weight_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.lstm_weight = lstm_weight
        self.arima_weight = 1.0 - lstm_weight
        self.forecast_horizon = forecast_horizon
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.arima_fitted = False
    
    def fit_arima(self, X: np.ndarray, y: np.ndarray):
        self.arima_model.fit(X, y)
        self.arima_fitted = True
    
    def forward(self, x: torch.Tensor, X_numpy: np.ndarray = None) -> torch.Tensor:
        lstm_pred = self.lstm_model(x)
        
        component_outputs = [lstm_pred]
        hidden_states = []
        
        with torch.no_grad():
            lstm_out, _ = self.lstm_model.lstm(x)
            last_hidden = lstm_out[:, -1, :]
            hidden_states.append(last_hidden)
        
        if self.tcn_model is not None:
            tcn_pred = self.tcn_model(x)
            component_outputs.append(tcn_pred)
            with torch.no_grad():
                tcn_features = self.tcn_model.tcn(x.transpose(1, 2))
                tcn_last = tcn_features[:, :, -1].transpose(1, 2).mean(dim=1)
                hidden_states.append(tcn_last)
        
        if self.arima_fitted and X_numpy is not None:
            arima_pred = self.arima_model.predict(X_numpy, self.forecast_horizon)
            arima_pred = torch.from_numpy(arima_pred).float().to(x.device)
            component_outputs.append(arima_pred)
            hidden_states.append(last_hidden)
        
        if self.use_ensemble and len(component_outputs) > 1:
            hybrid_pred, weights = self.fusion(component_outputs, hidden_states)
            return hybrid_pred
        
        if self.arima_fitted and X_numpy is not None and len(component_outputs) > 1:
            adaptive_weight = self.weight_network(last_hidden)
            adaptive_weight = adaptive_weight.unsqueeze(-1).unsqueeze(-1)
            
            if self.tcn_model is not None:
                hybrid_pred = (adaptive_weight * lstm_pred + 
                             (1 - adaptive_weight) * 0.5 * (tcn_pred + arima_pred))
            else:
                hybrid_pred = (adaptive_weight * lstm_pred + 
                             (1 - adaptive_weight) * arima_pred)
            return hybrid_pred
        
        return lstm_pred
    
    def predict_hybrid(self, x: torch.Tensor, X_numpy: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            pred = self.forward(x, X_numpy)
            return pred.cpu().numpy()


class AdaptiveHybridLSTMARIMA(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_layers: int = 3, dropout: float = 0.3,
                 forecast_horizon: int = 6):
        super(AdaptiveHybridLSTMARIMA, self).__init__()
        
        self.lstm_model = LSTMForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            forecast_horizon=forecast_horizon,
            use_attention=True,
            use_residual=True
        )
        
        self.arima_model = ARIMAForecaster(order=(2, 1, 2))
        
        self.confidence_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.forecast_horizon = forecast_horizon
        self.input_size = input_size
        self.arima_fitted = False
    
    def fit_arima(self, X: np.ndarray, y: np.ndarray):
        self.arima_model.fit(X, y)
        self.arima_fitted = True
    
    def forward(self, x: torch.Tensor, X_numpy: np.ndarray = None) -> torch.Tensor:
        lstm_pred = self.lstm_model(x)
        
        if self.arima_fitted and X_numpy is not None:
            with torch.no_grad():
                lstm_out, (h_n, _) = self.lstm_model.lstm(x)
                last_hidden = lstm_out[:, -1, :]
                
                confidence = self.confidence_network(last_hidden)
                
                arima_pred = self.arima_model.predict(X_numpy, self.forecast_horizon)
                arima_pred = torch.from_numpy(arima_pred).float().to(x.device)
                
                hybrid_pred = (confidence * lstm_pred + 
                             (1 - confidence) * arima_pred)
                return hybrid_pred
        
        return lstm_pred
