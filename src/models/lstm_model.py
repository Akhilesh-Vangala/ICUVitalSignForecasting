import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super(TemporalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.out(attn_output), attn_weights


class FeatureAttention(nn.Module):
    def __init__(self, input_size: int, reduction: int = 16):
        super(FeatureAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_size, input_size // reduction),
            nn.ReLU(),
            nn.Linear(input_size // reduction, input_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        x_perm = x.permute(0, 2, 1)
        
        avg_out = self.fc(self.avg_pool(x_perm).squeeze(-1))
        max_out = self.fc(self.max_pool(x_perm).squeeze(-1))
        
        attention = (avg_out + max_out).unsqueeze(-1)
        return x * attention.permute(0, 2, 1)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.2):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x).transpose(1, 2)).transpose(1, 2))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out).transpose(1, 2)).transpose(1, 2)
        return F.relu(out + residual)


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, 
                 num_layers: int = 3, dropout: float = 0.3,
                 forecast_horizon: int = 6, use_attention: bool = True,
                 use_residual: bool = True, bidirectional: bool = False):
        super(LSTMForecaster, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.bidirectional = bidirectional
        
        self.feature_attention = FeatureAttention(input_size) if use_attention else None
        
        lstm_hidden = hidden_size * 2 if bidirectional else hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.temporal_attention = TemporalAttention(lstm_hidden) if use_attention else None
        
        if use_residual:
            self.residual_blocks = nn.ModuleList([
                ResidualBlock(lstm_hidden, dropout) for _ in range(2)
            ])
        
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_hidden, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size, input_size * forecast_horizon)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_attention:
            x = self.feature_attention(x)
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        if self.temporal_attention:
            attn_out, attn_weights = self.temporal_attention(lstm_out)
            last_hidden = attn_out[:, -1, :]
        else:
            last_hidden = lstm_out[:, -1, :]
        
        if self.use_residual:
            last_hidden_expanded = last_hidden.unsqueeze(1)
            for residual_block in self.residual_blocks:
                last_hidden_expanded = residual_block(last_hidden_expanded)
            last_hidden = last_hidden_expanded.squeeze(1)
        
        output = self.fc_layers(last_hidden)
        output = output.view(-1, self.forecast_horizon, self.input_size)
        return output


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_layers: int = 3, dropout: float = 0.3,
                 forecast_horizon: int = 6):
        super(BidirectionalLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forecast_horizon = forecast_horizon
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        self.attention = TemporalAttention(hidden_size * 2)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size, input_size * forecast_horizon)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out)
        last_hidden = attn_out[:, -1, :]
        
        output = self.fc_layers(last_hidden)
        output = output.view(-1, self.forecast_horizon, self.input_size)
        return output
