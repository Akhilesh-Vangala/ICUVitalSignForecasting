"""XGBoost model for vital sign forecasting."""

import numpy as np
import xgboost as xgb
from typing import Tuple


class XGBoostForecaster:
    """XGBoost model for vital sign forecasting."""
    
    def __init__(self, n_estimators: int = 200, max_depth: int = 6,
                 learning_rate: float = 0.1, forecast_horizon: int = 6):
        """
        Initialize XGBoost model.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            forecast_horizon: Steps ahead to forecast
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.forecast_horizon = forecast_horizon
        self.models = {}
    
    def _reshape_for_xgboost(self, X: np.ndarray) -> np.ndarray:
        """Reshape sequences for XGBoost (flatten time dimension)."""
        n_samples, seq_len, n_features = X.shape
        return X.reshape(n_samples, seq_len * n_features)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostForecaster':
        """
        Fit XGBoost models for each vital sign and time step.
        
        Args:
            X: Input sequences (n_samples, seq_len, n_features)
            y: Target sequences (n_samples, forecast_horizon, n_features)
        """
        X_flat = self._reshape_for_xgboost(X)
        n_features = y.shape[2]
        
        for feature_idx in range(n_features):
            self.models[feature_idx] = {}
            for step in range(self.forecast_horizon):
                y_target = y[:, step, feature_idx]
                
                model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=42
                )
                
                model.fit(X_flat, y_target)
                self.models[feature_idx][step] = model
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate forecasts.
        
        Args:
            X: Input sequences (n_samples, seq_len, n_features)
            
        Returns:
            Forecasts (n_samples, forecast_horizon, n_features)
        """
        X_flat = self._reshape_for_xgboost(X)
        n_samples = X.shape[0]
        n_features = len(self.models)
        
        predictions = np.zeros((n_samples, self.forecast_horizon, n_features))
        
        for feature_idx in range(n_features):
            for step in range(self.forecast_horizon):
                pred = self.models[feature_idx][step].predict(X_flat)
                predictions[:, step, feature_idx] = pred
        
        return predictions
