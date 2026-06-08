import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
from typing import Dict, Optional, List
import sys
sys.path.append(str(Path(__file__).parent))
from utils import set_seed, count_parameters, save_model_checkpoint

from models.lstm_model import LSTMForecaster, BidirectionalLSTM
from models.hybrid_model import HybridLSTMARIMA, AdaptiveHybridLSTMARIMA
from models.transformer_model import TransformerForecaster
from models.tcn_model import TCNForecaster
from models.xgboost_model import XGBoostForecaster
from models.arima_model import ARIMAForecaster


class Lookahead:
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        self.slow_weights = {param: param.data.clone() for param in optimizer.param_groups[0]['params']}
    
    def step(self):
        self.optimizer.step()
        self.step_count += 1
        
        if self.step_count % self.k == 0:
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p in self.slow_weights:
                        self.slow_weights[p] += self.alpha * (p.data - self.slow_weights[p])
                        p.data.copy_(self.slow_weights[p])
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


class CosineAnnealingWarmRestarts(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * 
                (1 + np.cos(np.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = epoch
        else:
            self.T_cur = epoch
        
        if epoch >= self.T_i:
            self.T_cur = 0
            self.T_i *= self.T_mult
        
        super(CosineAnnealingWarmRestarts, self).step(epoch)


class ModelTrainer:
    def __init__(self, device: str = None, model_dir: str = 'models', seed: int = 42):
        set_seed(seed)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using device: {self.device}")
    
    def train_lstm(self, train_data: Dict, val_data: Dict,
                   epochs: int = 50, batch_size: int = 32,
                   learning_rate: float = 0.001, use_mixed_precision: bool = True,
                   use_lookahead: bool = True, model_name: str = 'lstm',
                   hidden_size: int = 256, num_layers: int = 3, dropout: float = 0.3) -> Dict:
        X_train = torch.FloatTensor(train_data['X']).to(self.device)
        y_train = torch.FloatTensor(train_data['y']).to(self.device)
        X_val = torch.FloatTensor(val_data['X']).to(self.device)
        y_val = torch.FloatTensor(val_data['y']).to(self.device)
        
        input_size = X_train.shape[2]
        forecast_horizon = y_train.shape[1]
        
        model = LSTMForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            forecast_horizon=forecast_horizon,
            use_attention=True,
            use_residual=True,
            bidirectional=False
        ).to(self.device)
        
        n_params = count_parameters(model)
        print(f"Model parameters: {n_params:,}")
        
        criterion = nn.MSELoss()
        base_optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        optimizer = Lookahead(base_optimizer) if use_lookahead else base_optimizer
        
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        scaler = GradScaler() if use_mixed_precision and self.device == 'cuda' else None
        
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        for epoch in tqdm(range(epochs), desc='Training LSTM'):
            model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                if use_mixed_precision and scaler is not None:
                    with autocast():
                        pred = model(X_batch)
                        loss = criterion(pred, y_batch)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(X_batch)
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                if use_lookahead:
                    optimizer.step()
                
                train_loss += loss.item()
            
            scheduler.step()
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val).item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(optimizer.optimizer.param_groups[0]['lr'] if use_lookahead else optimizer.param_groups[0]['lr'])
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), self.model_dir / f'{model_name}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        model.load_state_dict(torch.load(self.model_dir / f'{model_name}_best.pth'))
        return {'model': model, 'history': history}
    
    def train_hybrid(self, train_data: Dict, val_data: Dict,
                    epochs: int = 50, batch_size: int = 32,
                    learning_rate: float = 0.001, use_mixed_precision: bool = True,
                    use_lookahead: bool = True, model_name: str = 'hybrid_lstm_arima',
                    hidden_size: int = 256, num_layers: int = 3, dropout: float = 0.3,
                    use_tcn: bool = True) -> Dict:
        X_train = torch.FloatTensor(train_data['X']).to(self.device)
        y_train = torch.FloatTensor(train_data['y']).to(self.device)
        X_val = torch.FloatTensor(val_data['X']).to(self.device)
        y_val = torch.FloatTensor(val_data['y']).to(self.device)
        
        input_size = X_train.shape[2]
        forecast_horizon = y_train.shape[1]
        
        model = HybridLSTMARIMA(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            forecast_horizon=forecast_horizon,
            lstm_weight=0.7,
            use_attention=True,
            use_tcn=use_tcn,
            use_ensemble=True
        ).to(self.device)
        
        n_params = count_parameters(model)
        print(f"Model parameters: {n_params:,}")
        
        print("Fitting ARIMA component...")
        model.fit_arima(train_data['X'], train_data['y'])
        
        criterion = nn.MSELoss()
        base_optimizer = optim.AdamW(model.lstm_model.parameters(), lr=learning_rate, weight_decay=1e-4)
        if use_tcn:
            base_optimizer = optim.AdamW(
                list(model.lstm_model.parameters()) + list(model.tcn_model.parameters()),
                lr=learning_rate, weight_decay=1e-4
            )
        optimizer = Lookahead(base_optimizer) if use_lookahead else base_optimizer
        
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        scaler = GradScaler() if use_mixed_precision and self.device == 'cuda' else None
        
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        for epoch in tqdm(range(epochs), desc='Training Hybrid'):
            model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                X_batch_np = X_batch.cpu().numpy()
                
                if use_mixed_precision and scaler is not None:
                    with autocast():
                        pred = model(X_batch, X_batch_np)
                        loss = criterion(pred, y_batch)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(X_batch, X_batch_np)
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                if use_lookahead:
                    optimizer.step()
                
                train_loss += loss.item()
            
            scheduler.step()
            model.eval()
            with torch.no_grad():
                X_val_np = X_val.cpu().numpy()
                val_pred = model(X_val, X_val_np)
                val_loss = criterion(val_pred, y_val).item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(optimizer.optimizer.param_groups[0]['lr'] if use_lookahead else optimizer.param_groups[0]['lr'])
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'arima_fitted': True
                }, self.model_dir / f'{model_name}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        checkpoint = torch.load(self.model_dir / f'{model_name}_best.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        return {'model': model, 'history': history}
    
    def train_transformer(self, train_data: Dict, val_data: Dict,
                         epochs: int = 50, batch_size: int = 32,
                         learning_rate: float = 0.001, use_mixed_precision: bool = True,
                         use_lookahead: bool = True, model_name: str = 'transformer',
                         d_model: int = 256, nhead: int = 8, num_layers: int = 6) -> Dict:
        X_train = torch.FloatTensor(train_data['X']).to(self.device)
        y_train = torch.FloatTensor(train_data['y']).to(self.device)
        X_val = torch.FloatTensor(val_data['X']).to(self.device)
        y_val = torch.FloatTensor(val_data['y']).to(self.device)
        
        input_size = X_train.shape[2]
        forecast_horizon = y_train.shape[1]
        
        model = TransformerForecaster(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            forecast_horizon=forecast_horizon
        ).to(self.device)
        
        criterion = nn.MSELoss()
        base_optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        optimizer = Lookahead(base_optimizer) if use_lookahead else base_optimizer
        
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        scaler = GradScaler() if use_mixed_precision and self.device == 'cuda' else None
        
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        for epoch in tqdm(range(epochs), desc='Training Transformer'):
            model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                if use_mixed_precision and scaler is not None:
                    with autocast():
                        pred = model(X_batch)
                        loss = criterion(pred, y_batch)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(X_batch)
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                if use_lookahead:
                    optimizer.step()
                
                train_loss += loss.item()
            
            scheduler.step()
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val).item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(optimizer.optimizer.param_groups[0]['lr'] if use_lookahead else optimizer.param_groups[0]['lr'])
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), self.model_dir / f'{model_name}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        model.load_state_dict(torch.load(self.model_dir / f'{model_name}_best.pth'))
        return {'model': model, 'history': history}
    
    def train_tcn(self, train_data: Dict, val_data: Dict,
                  epochs: int = 50, batch_size: int = 32,
                  learning_rate: float = 0.001, use_mixed_precision: bool = True,
                  use_lookahead: bool = True, model_name: str = 'tcn') -> Dict:
        X_train = torch.FloatTensor(train_data['X']).to(self.device)
        y_train = torch.FloatTensor(train_data['y']).to(self.device)
        X_val = torch.FloatTensor(val_data['X']).to(self.device)
        y_val = torch.FloatTensor(val_data['y']).to(self.device)
        
        input_size = X_train.shape[2]
        forecast_horizon = y_train.shape[1]
        
        model = TCNForecaster(
            input_size=input_size,
            num_channels=[64, 128, 256],
            kernel_size=3,
            dropout=0.3,
            forecast_horizon=forecast_horizon
        ).to(self.device)
        
        criterion = nn.MSELoss()
        base_optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        optimizer = Lookahead(base_optimizer) if use_lookahead else base_optimizer
        
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        scaler = GradScaler() if use_mixed_precision and self.device == 'cuda' else None
        
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        for epoch in tqdm(range(epochs), desc='Training TCN'):
            model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                if use_mixed_precision and scaler is not None:
                    with autocast():
                        pred = model(X_batch)
                        loss = criterion(pred, y_batch)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(X_batch)
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                if use_lookahead:
                    optimizer.step()
                
                train_loss += loss.item()
            
            scheduler.step()
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val).item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(optimizer.optimizer.param_groups[0]['lr'] if use_lookahead else optimizer.param_groups[0]['lr'])
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), self.model_dir / f'{model_name}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        model.load_state_dict(torch.load(self.model_dir / f'{model_name}_best.pth'))
        return {'model': model, 'history': history}
    
    def train_xgboost(self, train_data: Dict, val_data: Dict,
                     model_name: str = 'xgboost') -> Dict:
        print("Training XGBoost...")
        model = XGBoostForecaster(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            forecast_horizon=train_data['y'].shape[1]
        )
        
        model.fit(train_data['X'], train_data['y'])
        
        import pickle
        with open(self.model_dir / f'{model_name}_best.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        return {'model': model}
    
    def train_arima(self, train_data: Dict, val_data: Dict,
                   model_name: str = 'arima') -> Dict:
        print("Training ARIMA...")
        model = ARIMAForecaster(order=(2, 1, 2))
        model.fit(train_data['X'], train_data['y'])
        
        import pickle
        with open(self.model_dir / f'{model_name}_best.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        return {'model': model}
