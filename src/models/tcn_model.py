import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, dilation: int, padding: int, dropout: float = 0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
    
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs: int, num_channels: list, kernel_size: int = 2, dropout: float = 0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class TCNForecaster(nn.Module):
    def __init__(self, input_size: int, num_channels: list = [64, 128, 256], 
                 kernel_size: int = 3, dropout: float = 0.3, forecast_horizon: int = 6):
        super(TCNForecaster, self).__init__()
        
        self.input_size = input_size
        self.forecast_horizon = forecast_horizon
        
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=num_channels[-1],
            num_heads=8,
            dropout=dropout,
            batch_first=False
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] * 2),
            nn.LayerNorm(num_channels[-1] * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] * 2, num_channels[-1]),
            nn.LayerNorm(num_channels[-1]),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(num_channels[-1], input_size * forecast_horizon)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, features = x.shape
        x = x.transpose(1, 2)
        
        tcn_out = self.tcn(x)
        tcn_out = tcn_out.transpose(1, 2)
        
        tcn_out = tcn_out.transpose(0, 1)
        attn_out, _ = self.attention(tcn_out, tcn_out, tcn_out)
        attn_out = attn_out.transpose(0, 1)
        
        last_hidden = attn_out[:, -1, :]
        output = self.fc_layers(last_hidden)
        output = output.view(batch_size, self.forecast_horizon, self.input_size)
        return output
