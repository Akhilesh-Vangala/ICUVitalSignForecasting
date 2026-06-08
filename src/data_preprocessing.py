"""Data preprocessing for MIMIC-III ICU vital signs."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class MIMICDataPreprocessor:
    """Preprocess MIMIC-III data for vital sign forecasting."""
    
    def __init__(self, data_path: str, forecast_horizon: int = 6):
        """
        Initialize preprocessor.
        
        Args:
            data_path: Path to MIMIC-III data directory
            forecast_horizon: Hours ahead to forecast (default: 6)
        """
        self.data_path = Path(data_path)
        self.forecast_horizon = forecast_horizon
        self.vital_signs = ['heartrate', 'sysbp', 'diabp', 'temp', 'resprate', 'spo2']
        
    def load_mimic_data(self) -> pd.DataFrame:
        """Load MIMIC-III chartevents and patients data."""
        try:
            chartevents = pd.read_csv(self.data_path / 'CHARTEVENTS.csv', low_memory=False)
            patients = pd.read_csv(self.data_path / 'PATIENTS.csv', low_memory=False)
            icustays = pd.read_csv(self.data_path / 'ICUSTAYS.csv', low_memory=False)
            
            chartevents = chartevents.merge(icustays[['ICUSTAY_ID', 'SUBJECT_ID', 'INTIME', 'OUTTIME']], 
                                          on='ICUSTAY_ID', how='left')
            chartevents = chartevents.merge(patients[['SUBJECT_ID', 'GENDER', 'DOB']], 
                                          on='SUBJECT_ID', how='left')
            
            return chartevents
        except FileNotFoundError:
            print("MIMIC-III files not found. Generating synthetic data for testing...")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self, n_patients: int = 40000) -> pd.DataFrame:
        """Generate synthetic ICU vital signs data matching MIMIC-III scale."""
        import random
        random.seed(42)
        np.random.seed(42)
        
        data = []
        for patient_id in range(1, n_patients + 1):
            stay_duration = np.random.randint(24, 240)
            base_hr = np.random.normal(80, 15)
            base_sbp = np.random.normal(120, 20)
            
            for hour in range(stay_duration):
                trend = 0.1 * np.sin(hour / 24.0)
                noise_hr = np.random.normal(0, 3)
                noise_bp = np.random.normal(0, 5)
                
                data.append({
                    'SUBJECT_ID': patient_id,
                    'ICUSTAY_ID': patient_id,
                    'CHARTTIME': pd.Timestamp('2020-01-01') + pd.Timedelta(hours=hour),
                    'heartrate': np.clip(base_hr + trend + noise_hr, 30, 200),
                    'sysbp': np.clip(base_sbp + trend * 2 + noise_bp, 40, 250),
                    'diabp': np.clip(base_sbp * 0.6 + noise_bp * 0.8, 30, 150),
                    'temp': np.clip(np.random.normal(37.0, 0.5), 30, 42),
                    'resprate': np.clip(np.random.normal(18, 4), 8, 40),
                    'spo2': np.clip(np.random.normal(98, 2), 70, 100),
                    'GENDER': np.random.choice(['M', 'F']),
                    'AGE': np.random.randint(18, 90)
                })
        
        return pd.DataFrame(data)
    
    def extract_vital_signs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and clean vital signs from chartevents.
        
        Args:
            df: Raw chartevents dataframe
            
        Returns:
            Cleaned vital signs dataframe
        """
        vital_df = df.copy()
        
        if 'CHARTTIME' in vital_df.columns:
            vital_df['CHARTTIME'] = pd.to_datetime(vital_df['CHARTTIME'])
            vital_df = vital_df.sort_values(['SUBJECT_ID', 'ICUSTAY_ID', 'CHARTTIME'])
        
        for vs in self.vital_signs:
            if vs not in vital_df.columns:
                continue
            
            vital_df[vs] = pd.to_numeric(vital_df[vs], errors='coerce')
            
            if vs == 'heartrate':
                vital_df[vs] = vital_df[vs].clip(30, 200)
            elif vs in ['sysbp', 'diabp']:
                vital_df[vs] = vital_df[vs].clip(40, 250)
            elif vs == 'temp':
                vital_df[vs] = vital_df[vs].clip(30, 42)
            elif vs == 'resprate':
                vital_df[vs] = vital_df[vs].clip(8, 40)
            elif vs == 'spo2':
                vital_df[vs] = vital_df[vs].clip(70, 100)
        
        return vital_df
    
    def create_time_series(self, df: pd.DataFrame, 
                          window_size: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time series sequences for forecasting.
        
        Args:
            df: Vital signs dataframe
            window_size: Historical window size in hours
            
        Returns:
            X: Input sequences (n_samples, window_size, n_features)
            y: Target values (n_samples, forecast_horizon, n_features)
        """
        sequences = []
        targets = []
        
        for stay_id in df['ICUSTAY_ID'].unique():
            stay_data = df[df['ICUSTAY_ID'] == stay_id].copy()
            stay_data = stay_data.sort_values('CHARTTIME')
            
            vital_cols = [col for col in self.vital_signs if col in stay_data.columns]
            if len(vital_cols) == 0:
                continue
            
            stay_data = stay_data[['CHARTTIME'] + vital_cols].dropna()
            
            if len(stay_data) < window_size + self.forecast_horizon:
                continue
            
            for i in range(len(stay_data) - window_size - self.forecast_horizon + 1):
                window = stay_data.iloc[i:i+window_size][vital_cols].values
                target = stay_data.iloc[i+window_size:i+window_size+self.forecast_horizon][vital_cols].values
                
                if not np.isnan(window).any() and not np.isnan(target).any():
                    sequences.append(window)
                    targets.append(target)
        
        X = np.array(sequences)
        y = np.array(targets)
        
        return X, y
    
    def normalize_data(self, X: np.ndarray, y: np.ndarray, 
                      fit: bool = True) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Normalize vital signs data.
        
        Args:
            X: Input sequences
            y: Target sequences
            fit: Whether to fit normalization parameters
            
        Returns:
            Normalized X, y, and normalization parameters
        """
        if fit:
            self.mean = X.mean(axis=(0, 1), keepdims=True)
            self.std = X.std(axis=(0, 1), keepdims=True) + 1e-8
        
        X_norm = (X - self.mean) / self.std
        y_norm = (y - self.mean) / self.std
        
        norm_params = {'mean': self.mean, 'std': self.std}
        
        return X_norm, y_norm, norm_params
    
    def preprocess(self, window_size: int = 24, 
                   train_split: float = 0.7,
                   val_split: float = 0.15) -> dict:
        """
        Complete preprocessing pipeline.
        
        Args:
            window_size: Historical window size
            train_split: Training data proportion
            val_split: Validation data proportion
            
        Returns:
            Dictionary with preprocessed data splits
        """
        print("Loading MIMIC-III data...")
        raw_df = self.load_mimic_data()
        
        print("Extracting vital signs...")
        vital_df = self.extract_vital_signs(raw_df)
        
        print("Creating time series sequences...")
        X, y = self.create_time_series(vital_df, window_size)
        
        print(f"Created {len(X)} sequences")
        
        n_samples = len(X)
        n_train = int(n_samples * train_split)
        n_val = int(n_samples * val_split)
        
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_val = X[n_train:n_train+n_val]
        y_val = y[n_train:n_train+n_val]
        X_test = X[n_train+n_val:]
        y_test = y[n_train+n_val:]
        
        print("Normalizing data...")
        X_train_norm, y_train_norm, norm_params = self.normalize_data(X_train, y_train, fit=True)
        X_val_norm, y_val_norm, _ = self.normalize_data(X_val, y_val, fit=False)
        X_test_norm, y_test_norm, _ = self.normalize_data(X_test, y_test, fit=False)
        
        return {
            'train': {'X': X_train_norm, 'y': y_train_norm},
            'val': {'X': X_val_norm, 'y': y_val_norm},
            'test': {'X': X_test_norm, 'y': y_test_norm},
            'norm_params': norm_params,
            'vital_signs': self.vital_signs
        }


if __name__ == '__main__':
    preprocessor = MIMICDataPreprocessor(data_path='data/raw', forecast_horizon=6)
    data = preprocessor.preprocess(window_size=24)
    
    print(f"Training samples: {data['train']['X'].shape}")
    print(f"Validation samples: {data['val']['X'].shape}")
    print(f"Test samples: {data['test']['X'].shape}")
