import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class ClinicalMetrics:
    def __init__(self):
        self.deterioration_thresholds = {
            'heartrate': {'low': 40, 'high': 160},
            'sysbp': {'low': 70, 'high': 200},
            'diabp': {'low': 40, 'high': 120},
            'temp': {'low': 36.0, 'high': 38.5},
            'resprate': {'low': 9, 'high': 24},
            'spo2': {'low': 90, 'high': 100}
        }
    
    def detect_deterioration(self, predictions: np.ndarray, vital_signs: List[str]) -> np.ndarray:
        n_samples, forecast_horizon, n_features = predictions.shape
        deterioration_flags = np.zeros((n_samples, forecast_horizon), dtype=bool)
        
        for i, vs in enumerate(vital_signs):
            if vs.lower() in self.deterioration_thresholds:
                thresholds = self.deterioration_thresholds[vs.lower()]
                for t in range(forecast_horizon):
                    low_flag = predictions[:, t, i] < thresholds['low']
                    high_flag = predictions[:, t, i] > thresholds['high']
                    deterioration_flags[:, t] |= (low_flag | high_flag)
        
        return deterioration_flags
    
    def calculate_early_warning_score(self, predictions: np.ndarray, vital_signs: List[str]) -> np.ndarray:
        n_samples, forecast_horizon, n_features = predictions.shape
        scores = np.zeros((n_samples, forecast_horizon))
        
        for i, vs in enumerate(vital_signs):
            if vs.lower() in self.deterioration_thresholds:
                thresholds = self.deterioration_thresholds[vs.lower()]
                for t in range(forecast_horizon):
                    values = predictions[:, t, i]
                    low_penalty = np.maximum(0, (thresholds['low'] - values) / thresholds['low']) * 3
                    high_penalty = np.maximum(0, (values - thresholds['high']) / thresholds['high']) * 3
                    scores[:, t] += (low_penalty + high_penalty)
        
        return scores


class ModelEvaluator:
    def __init__(self, norm_params: Dict = None):
        self.norm_params = norm_params
        self.clinical_metrics = ClinicalMetrics()
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        if self.norm_params is None:
            return data
        mean = self.norm_params['mean']
        std = self.norm_params['std']
        return data * std + mean
    
    def calculate_uncertainty(self, predictions: np.ndarray, num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        mean_pred = np.mean(predictions, axis=0) if len(predictions.shape) > 2 else predictions
        std_pred = np.std(predictions, axis=0) if len(predictions.shape) > 2 else np.zeros_like(mean_pred)
        return mean_pred, std_pred
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100
        r2 = r2_score(y_true_flat, y_pred_flat)
        
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        mape_median = np.median(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100
        
        correlation = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'MAPE_median': mape_median,
            'R2': r2,
            'MSE': mse,
            'Correlation': correlation
        }
    
    def calculate_clinical_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   vital_signs: List[str]) -> Dict:
        y_true_denorm = self.denormalize(y_true)
        y_pred_denorm = self.denormalize(y_pred)
        
        true_deterioration = self.clinical_metrics.detect_deterioration(y_true_denorm, vital_signs)
        pred_deterioration = self.clinical_metrics.detect_deterioration(y_pred_denorm, vital_signs)
        
        true_ews = self.clinical_metrics.calculate_early_warning_score(y_true_denorm, vital_signs)
        pred_ews = self.clinical_metrics.calculate_early_warning_score(y_pred_denorm, vital_signs)
        
        tp = np.sum((true_deterioration & pred_deterioration))
        fp = np.sum((~true_deterioration & pred_deterioration))
        fn = np.sum((true_deterioration & ~pred_deterioration))
        tn = np.sum((~true_deterioration & ~pred_deterioration))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        ews_rmse = np.sqrt(mean_squared_error(true_ews.flatten(), pred_ews.flatten()))
        ews_mae = mean_absolute_error(true_ews.flatten(), pred_ews.flatten())
        
        return {
            'Deterioration_TP': tp,
            'Deterioration_FP': fp,
            'Deterioration_FN': fn,
            'Deterioration_TN': tn,
            'Deterioration_Precision': precision,
            'Deterioration_Recall': recall,
            'Deterioration_F1': f1,
            'EWS_RMSE': ews_rmse,
            'EWS_MAE': ews_mae
        }
    
    def calculate_per_vital_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    vital_signs: List[str]) -> pd.DataFrame:
        y_true_denorm = self.denormalize(y_true)
        y_pred_denorm = self.denormalize(y_pred)
        
        results = []
        for i, vs in enumerate(vital_signs):
            true_vals = y_true_denorm[:, :, i].flatten()
            pred_vals = y_pred_denorm[:, :, i].flatten()
            
            rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
            mae = mean_absolute_error(true_vals, pred_vals)
            mape = np.mean(np.abs((true_vals - pred_vals) / (true_vals + 1e-8))) * 100
            r2 = r2_score(true_vals, pred_vals)
            correlation = np.corrcoef(true_vals, pred_vals)[0, 1]
            
            results.append({
                'Vital_Sign': vs,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'R2': r2,
                'Correlation': correlation
            })
        
        return pd.DataFrame(results)
    
    def evaluate_model(self, model, test_data: Dict, 
                      model_type: str = 'pytorch',
                      device: str = 'cpu',
                      vital_signs: List[str] = None,
                      use_uncertainty: bool = False) -> Dict:
        X_test = test_data['X']
        y_test = test_data['y']
        
        if model_type == 'pytorch':
            model.eval()
            X_tensor = torch.FloatTensor(X_test).to(device)
            
            with torch.no_grad():
                if hasattr(model, 'predict_hybrid'):
                    predictions = model.predict_hybrid(X_tensor, X_test)
                else:
                    if use_uncertainty:
                        preds = []
                        for _ in range(10):
                            pred_tensor = model(X_tensor)
                            preds.append(pred_tensor.cpu().numpy())
                        predictions = np.array(preds)
                    else:
                        pred_tensor = model(X_tensor)
                        predictions = pred_tensor.cpu().numpy()
        
        elif model_type == 'xgboost':
            predictions = model.predict(X_test)
        
        elif model_type == 'arima':
            predictions = model.predict(X_test, forecast_horizon=y_test.shape[1])
        
        if use_uncertainty and len(predictions.shape) == 4:
            mean_pred, std_pred = self.calculate_uncertainty(predictions)
            predictions = mean_pred
        
        y_true_denorm = self.denormalize(y_test)
        y_pred_denorm = self.denormalize(predictions)
        
        metrics = self.calculate_metrics(y_true_denorm, y_pred_denorm)
        
        result = {
            'metrics': metrics,
            'predictions': y_pred_denorm,
            'true_values': y_true_denorm
        }
        
        if vital_signs is not None:
            clinical_metrics = self.calculate_clinical_metrics(y_true_denorm, y_pred_denorm, vital_signs)
            per_vital_metrics = self.calculate_per_vital_metrics(y_true_denorm, y_pred_denorm, vital_signs)
            result['clinical_metrics'] = clinical_metrics
            result['per_vital_metrics'] = per_vital_metrics
        
        return result
    
    def compare_models(self, results: Dict[str, Dict]) -> pd.DataFrame:
        comparison = []
        for model_name, result in results.items():
            metrics = result['metrics']
            row = {
                'Model': model_name,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'MAPE': metrics['MAPE'],
                'R2': metrics['R2'],
                'Correlation': metrics['Correlation']
            }
            
            if 'clinical_metrics' in result:
                clinical = result['clinical_metrics']
                row['Deterioration_F1'] = clinical['Deterioration_F1']
                row['Deterioration_Recall'] = clinical['Deterioration_Recall']
                row['EWS_RMSE'] = clinical['EWS_RMSE']
            
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        return df.sort_values('RMSE')
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                        vital_signs: List[str], save_path: str = None):
        n_features = len(vital_signs)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, vs_name in enumerate(vital_signs):
            if i >= n_features:
                break
            
            true_vals = y_true[:, :, i].flatten()
            pred_vals = y_pred[:, :, i].flatten()
            
            axes[i].scatter(true_vals, pred_vals, alpha=0.5, s=10)
            min_val = min(true_vals.min(), pred_vals.min())
            max_val = max(true_vals.max(), pred_vals.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            axes[i].set_xlabel(f'True {vs_name}')
            axes[i].set_ylabel(f'Predicted {vs_name}')
            axes[i].set_title(f'{vs_name} Forecast')
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_forecast_trajectory(self, y_true: np.ndarray, y_pred: np.ndarray,
                                vital_signs: List[str], sample_idx: int = 0,
                                save_path: str = None, uncertainty: np.ndarray = None):
        n_features = len(vital_signs)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        forecast_horizon = y_true.shape[1]
        time_steps = np.arange(forecast_horizon)
        
        for i, vs_name in enumerate(vital_signs):
            if i >= n_features:
                break
            
            true_traj = y_true[sample_idx, :, i]
            pred_traj = y_pred[sample_idx, :, i]
            
            axes[i].plot(time_steps, true_traj, 'o-', label='True', linewidth=2, markersize=6)
            axes[i].plot(time_steps, pred_traj, 's-', label='Predicted', linewidth=2, markersize=6)
            
            if uncertainty is not None:
                std_traj = uncertainty[sample_idx, :, i]
                axes[i].fill_between(time_steps, pred_traj - std_traj, pred_traj + std_traj,
                                    alpha=0.3, label='Uncertainty')
            
            axes[i].set_xlabel('Hours Ahead')
            axes[i].set_ylabel(vs_name)
            axes[i].set_title(f'{vs_name} Forecast Trajectory')
            axes[i].legend()
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
