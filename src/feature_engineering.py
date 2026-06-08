import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.fft import fft, fftfreq
import pywt
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class ClinicalScoreCalculator:
    def __init__(self):
        pass
    
    def calculate_sofa_component(self, vital_signs: np.ndarray) -> np.ndarray:
        hr, sbp, dbp, temp, rr, spo2 = vital_signs.T
        sofa_scores = np.zeros(len(hr))
        
        sofa_scores += np.where(hr < 40, 3, np.where(hr > 160, 3, np.where(hr < 70, 2, 0)))
        sofa_scores += np.where(sbp < 70, 4, np.where(sbp < 100, 3, 0))
        sofa_scores += np.where(rr < 9, 4, np.where(rr > 24, 2, 0))
        sofa_scores += np.where(temp < 36, 2, np.where(temp > 38.5, 2, 0))
        sofa_scores += np.where(spo2 < 90, 4, np.where(spo2 < 95, 2, 0))
        
        return sofa_scores
    
    def calculate_apache_component(self, vital_signs: np.ndarray) -> np.ndarray:
        hr, sbp, dbp, temp, rr, spo2 = vital_signs.T
        apache_scores = np.zeros(len(hr))
        
        apache_scores += np.where(hr < 40, 4, np.where(hr > 180, 4, np.where(hr < 55, 3, np.where(hr > 140, 2, 0))))
        apache_scores += np.where(sbp < 50, 4, np.where(sbp < 70, 3, np.where(sbp > 160, 2, 0)))
        apache_scores += np.where(rr < 5, 4, np.where(rr > 50, 4, np.where(rr < 10, 2, np.where(rr > 35, 2, 0))))
        apache_scores += np.where(temp < 30, 4, np.where(temp > 41, 4, np.where(temp < 32, 3, np.where(temp > 39, 2, 0))))
        
        return apache_scores


class FeatureEngineer:
    def __init__(self):
        self.feature_names = []
        self.clinical_calc = ClinicalScoreCalculator()
    
    def extract_temporal_features(self, X: np.ndarray) -> np.ndarray:
        n_samples, seq_len, n_features = X.shape
        features = []
        
        for i in range(n_samples):
            sample_features = []
            for j in range(n_features):
                series = X[i, :, j]
                
                sample_features.extend([
                    np.mean(series),
                    np.std(series),
                    np.min(series),
                    np.max(series),
                    np.median(series),
                    np.percentile(series, 25),
                    np.percentile(series, 75),
                    stats.skew(series),
                    stats.kurtosis(series),
                    series[-1] - series[0],
                    np.mean(np.diff(series)),
                    np.std(np.diff(series)),
                    np.mean(np.abs(np.diff(series))),
                    np.sum(np.diff(series) > 0) / len(series),
                    np.sum(np.diff(series) < 0) / len(series),
                    np.corrcoef(np.arange(len(series)), series)[0, 1] if len(series) > 1 else 0
                ])
            
            features.append(sample_features)
        
        return np.array(features)
    
    def extract_trend_features(self, X: np.ndarray) -> np.ndarray:
        n_samples, seq_len, n_features = X.shape
        features = []
        
        for i in range(n_samples):
            sample_features = []
            for j in range(n_features):
                series = X[i, :, j]
                time_points = np.arange(len(series))
                
                slope, intercept = np.polyfit(time_points, series, 1)
                r_squared = np.corrcoef(time_points, series)[0, 1] ** 2 if len(series) > 1 else 0
                
                poly2 = np.polyfit(time_points, series, 2) if len(series) > 2 else [0, 0, 0]
                curvature = poly2[0]
                
                sample_features.extend([slope, intercept, r_squared, curvature])
            
            features.append(sample_features)
        
        return np.array(features)
    
    def extract_frequency_features(self, X: np.ndarray) -> np.ndarray:
        n_samples, seq_len, n_features = X.shape
        features = []
        
        for i in range(n_samples):
            sample_features = []
            for j in range(n_features):
                series = X[i, :, j]
                
                if len(series) < 4:
                    sample_features.extend([0, 0, 0, 0])
                    continue
                
                fft_vals = np.abs(fft(series))
                fft_freqs = fftfreq(len(series))
                
                dominant_freq_idx = np.argmax(fft_vals[1:len(fft_vals)//2]) + 1
                dominant_freq = abs(fft_freqs[dominant_freq_idx])
                dominant_power = fft_vals[dominant_freq_idx]
                
                total_power = np.sum(fft_vals[1:len(fft_vals)//2])
                spectral_centroid = np.sum(fft_freqs[1:len(fft_freqs)//2] * fft_vals[1:len(fft_vals)//2]) / (total_power + 1e-8)
                
                sample_features.extend([dominant_freq, dominant_power, total_power, spectral_centroid])
            
            features.append(sample_features)
        
        return np.array(features)
    
    def extract_wavelet_features(self, X: np.ndarray, wavelet: str = 'db4') -> np.ndarray:
        n_samples, seq_len, n_features = X.shape
        features = []
        
        for i in range(n_samples):
            sample_features = []
            for j in range(n_features):
                series = X[i, :, j]
                
                if len(series) < 8:
                    sample_features.extend([0] * 8)
                    continue
                
                try:
                    coeffs = pywt.wavedec(series, wavelet, level=min(3, int(np.log2(len(series)))))
                    for coeff in coeffs:
                        sample_features.extend([
                            np.mean(np.abs(coeff)),
                            np.std(coeff),
                            np.max(np.abs(coeff))
                        ])
                    while len(sample_features) < (j + 1) * 8:
                        sample_features.append(0)
                except:
                    sample_features.extend([0] * 8)
            
            features.append(sample_features[:n_features * 8])
        
        return np.array(features)
    
    def extract_interaction_features(self, X: np.ndarray) -> np.ndarray:
        n_samples, seq_len, n_features = X.shape
        features = []
        
        for i in range(n_samples):
            sample_features = []
            
            for j in range(n_features):
                for k in range(j + 1, n_features):
                    series1 = X[i, :, j]
                    series2 = X[i, :, k]
                    
                    correlation = np.corrcoef(series1, series2)[0, 1] if len(series1) > 1 else 0
                    ratio_mean = np.mean(series1) / (np.mean(series2) + 1e-8)
                    diff_mean = np.mean(series1) - np.mean(series2)
                    product_mean = np.mean(series1 * series2)
                    
                    sample_features.extend([correlation, ratio_mean, diff_mean, product_mean])
            
            features.append(sample_features)
        
        return np.array(features)
    
    def extract_clinical_features(self, X: np.ndarray) -> np.ndarray:
        n_samples, seq_len, n_features = X.shape
        features = []
        
        for i in range(n_samples):
            vital_signs = X[i, :, :]
            
            sofa_scores = self.clinical_calc.calculate_sofa_component(vital_signs)
            apache_scores = self.clinical_calc.calculate_apache_component(vital_signs)
            
            features.append([
                np.mean(sofa_scores),
                np.std(sofa_scores),
                np.max(sofa_scores),
                np.mean(apache_scores),
                np.std(apache_scores),
                np.max(apache_scores)
            ])
        
        return np.array(features)
    
    def extract_statistical_features(self, X: np.ndarray) -> np.ndarray:
        n_samples, seq_len, n_features = X.shape
        features = []
        
        for i in range(n_samples):
            sample_features = []
            for j in range(n_features):
                series = X[i, :, j]
                
                sample_features.extend([
                    stats.iqr(series),
                    stats.entropy(np.histogram(series, bins=10)[0] + 1e-10),
                    np.var(series),
                    stats.variation(series) if np.mean(series) != 0 else 0,
                    np.mean(np.abs(series - np.mean(series))),
                    np.sum(series > np.percentile(series, 90)) / len(series),
                    np.sum(series < np.percentile(series, 10)) / len(series)
                ])
            
            features.append(sample_features)
        
        return np.array(features)
    
    def create_engineered_features(self, X: np.ndarray) -> Tuple[np.ndarray, list]:
        temporal = self.extract_temporal_features(X)
        trend = self.extract_trend_features(X)
        frequency = self.extract_frequency_features(X)
        wavelet = self.extract_wavelet_features(X)
        interaction = self.extract_interaction_features(X)
        clinical = self.extract_clinical_features(X)
        statistical = self.extract_statistical_features(X)
        
        engineered = np.hstack([temporal, trend, frequency, wavelet, interaction, clinical, statistical])
        
        feature_names = []
        vital_signs = ['HR', 'SBP', 'DBP', 'TEMP', 'RR', 'SpO2']
        
        for vs in vital_signs:
            feature_names.extend([
                f'{vs}_mean', f'{vs}_std', f'{vs}_min', f'{vs}_max', f'{vs}_median',
                f'{vs}_q25', f'{vs}_q75', f'{vs}_skew', f'{vs}_kurtosis',
                f'{vs}_range', f'{vs}_diff_mean', f'{vs}_diff_std', f'{vs}_diff_abs_mean',
                f'{vs}_increasing_ratio', f'{vs}_decreasing_ratio', f'{vs}_trend_corr'
            ])
        
        for vs in vital_signs:
            feature_names.extend([f'{vs}_slope', f'{vs}_intercept', f'{vs}_r2', f'{vs}_curvature'])
        
        for vs in vital_signs:
            feature_names.extend([f'{vs}_dominant_freq', f'{vs}_dominant_power', f'{vs}_total_power', f'{vs}_spectral_centroid'])
        
        for vs in vital_signs:
            feature_names.extend([f'{vs}_wavelet_mean_1', f'{vs}_wavelet_std_1', f'{vs}_wavelet_max_1',
                                 f'{vs}_wavelet_mean_2', f'{vs}_wavelet_std_2', f'{vs}_wavelet_max_2',
                                 f'{vs}_wavelet_mean_3', f'{vs}_wavelet_std_3', f'{vs}_wavelet_max_3'])
        
        for j, vs1 in enumerate(vital_signs):
            for vs2 in vital_signs[j+1:]:
                feature_names.extend([
                    f'{vs1}_{vs2}_corr',
                    f'{vs1}_{vs2}_ratio',
                    f'{vs1}_{vs2}_diff',
                    f'{vs1}_{vs2}_product'
                ])
        
        feature_names.extend(['SOFA_mean', 'SOFA_std', 'SOFA_max', 'APACHE_mean', 'APACHE_std', 'APACHE_max'])
        
        for vs in vital_signs:
            feature_names.extend([
                f'{vs}_iqr', f'{vs}_entropy', f'{vs}_variance', f'{vs}_cv',
                f'{vs}_mad', f'{vs}_high_percentile_ratio', f'{vs}_low_percentile_ratio'
            ])
        
        return engineered, feature_names
