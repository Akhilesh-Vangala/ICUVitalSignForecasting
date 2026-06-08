"""ARIMA model for vital sign forecasting."""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


class ARIMAForecaster:
    """ARIMA model for vital sign forecasting."""
    
    def __init__(self, order: Tuple[int, int, int] = (2, 1, 2),
                 seasonal_order: Tuple[int, int, int, int] = None):
        """
        Initialize ARIMA model.
        
        Args:
            order: (p, d, q) ARIMA order
            seasonal_order: (P, D, Q, s) seasonal order
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.models = {}
    
    def _check_stationarity(self, series: np.ndarray) -> bool:
        """Check if time series is stationary."""
        result = adfuller(series)
        return result[1] <= 0.05
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ARIMAForecaster':
        """
        Fit ARIMA models for each vital sign.
        
        Args:
            X: Input sequences (n_samples, seq_len, n_features)
            y: Target sequences (n_samples, forecast_horizon, n_features)
        """
        n_features = X.shape[2]
        forecast_horizon = y.shape[1]
        
        for feature_idx in range(n_features):
            series = X[:, :, feature_idx].flatten()
            
            try:
                model = ARIMA(series, order=self.order, 
                            seasonal_order=self.seasonal_order)
                fitted_model = model.fit()
                self.models[feature_idx] = fitted_model
            except Exception as e:
                print(f"Warning: ARIMA fitting failed for feature {feature_idx}: {e}")
                self.models[feature_idx] = None
        
        return self
    
    def predict(self, X: np.ndarray, forecast_horizon: int = 6) -> np.ndarray:
        """
        Generate forecasts.
        
        Args:
            X: Input sequences (n_samples, seq_len, n_features)
            forecast_horizon: Steps ahead to forecast
            
        Returns:
            Forecasts (n_samples, forecast_horizon, n_features)
        """
        n_samples, seq_len, n_features = X.shape
        predictions = np.zeros((n_samples, forecast_horizon, n_features))
        
        for feature_idx in range(n_features):
            if feature_idx not in self.models or self.models[feature_idx] is None:
                last_values = X[:, -1, feature_idx]
                predictions[:, :, feature_idx] = np.tile(
                    last_values.reshape(-1, 1), (1, forecast_horizon)
                )
                continue
            
            for sample_idx in range(n_samples):
                series = X[sample_idx, :, feature_idx]
                
                try:
                    forecast = self.models[feature_idx].forecast(steps=forecast_horizon)
                    predictions[sample_idx, :, feature_idx] = forecast
                except Exception as e:
                    last_value = series[-1]
                    predictions[sample_idx, :, feature_idx] = last_value
        
        return predictions
    
    def auto_arima(self, X: np.ndarray, max_p: int = 3, max_d: int = 2, 
                   max_q: int = 3) -> 'ARIMAForecaster':
        """
        Automatically select ARIMA order using AIC.
        
        Args:
            X: Input sequences
            max_p: Maximum AR order
            max_d: Maximum differencing order
            max_q: Maximum MA order
        """
        n_features = X.shape[2]
        
        for feature_idx in range(n_features):
            series = X[:, :, feature_idx].flatten()
            best_aic = np.inf
            best_order = (1, 1, 1)
            
            for p in range(max_p + 1):
                for d in range(max_d + 1):
                    for q in range(max_q + 1):
                        try:
                            model = ARIMA(series, order=(p, d, q))
                            fitted = model.fit()
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            try:
                model = ARIMA(series, order=best_order)
                self.models[feature_idx] = model.fit()
            except:
                self.models[feature_idx] = None
        
        return self
