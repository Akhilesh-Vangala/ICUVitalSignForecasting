import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SHAPAnalyzer:
    def __init__(self, model, model_type: str = 'pytorch', device: str = 'cpu'):
        self.model = model
        self.model_type = model_type
        self.device = device
    
    def explain_pytorch_model(self, X_sample: np.ndarray, 
                             background_data: np.ndarray,
                             n_samples: int = 100) -> Dict:
        self.model.eval()
        
        background_subset = background_data[:n_samples]
        
        def model_wrapper(X):
            X_tensor = torch.FloatTensor(X).to(self.device)
            with torch.no_grad():
                pred = self.model(X_tensor)
            return pred.cpu().numpy().flatten()
        
        try:
            explainer = shap.DeepExplainer(
                model_wrapper,
                torch.FloatTensor(background_subset).to(self.device)
            )
            
            shap_values = explainer.shap_values(
                torch.FloatTensor(X_sample).to(self.device)
            )
            
            return {
                'shap_values': shap_values,
                'base_values': explainer.expected_value
            }
        except Exception as e:
            print(f"DeepExplainer failed, using GradientExplainer: {e}")
            explainer = shap.GradientExplainer(
                model_wrapper,
                torch.FloatTensor(background_subset).to(self.device)
            )
            
            shap_values = explainer.shap_values(
                torch.FloatTensor(X_sample).to(self.device)
            )
            
            return {
                'shap_values': shap_values,
                'base_values': np.mean(model_wrapper(background_subset))
            }
    
    def explain_xgboost_model(self, X_sample: np.ndarray,
                             background_data: np.ndarray) -> Dict:
        if hasattr(self.model, 'models') and len(self.model.models) > 0:
            first_model = list(self.model.models.values())[0]
            if len(first_model) > 0:
                explainer = shap.TreeExplainer(list(first_model.values())[0])
                
                X_flat_sample = X_sample.reshape(1, -1)
                X_flat_background = background_data.reshape(len(background_data), -1)
                
                shap_values = explainer.shap_values(X_flat_sample)
                
                return {
                    'shap_values': shap_values,
                    'base_values': explainer.expected_value
                }
        
        return {
            'shap_values': np.zeros(X_sample.shape),
            'base_values': 0.0
        }
    
    def plot_shap_summary(self, shap_values: np.ndarray,
                         feature_names: List[str],
                         save_path: str = None):
        if len(shap_values.shape) == 3:
            shap_values_flat = shap_values.reshape(-1, shap_values.shape[-1])
        else:
            shap_values_flat = shap_values
        
        try:
            shap.summary_plot(
                shap_values_flat,
                feature_names=feature_names,
                show=False,
                max_display=20
            )
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Summary plot failed: {e}")
    
    def plot_shap_waterfall(self, shap_values: np.ndarray,
                           base_value: float,
                           feature_names: List[str],
                           sample_idx: int = 0,
                           save_path: str = None):
        if len(shap_values.shape) == 3:
            shap_vals = shap_values[sample_idx].mean(axis=0)
        else:
            shap_vals = shap_values[sample_idx] if len(shap_values.shape) > 1 else shap_values
        
        try:
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_vals,
                    base_values=base_value,
                    data=np.zeros_like(shap_vals),
                    feature_names=feature_names
                ),
                show=False
            )
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Waterfall plot failed: {e}")
    
    def plot_temporal_shap(self, shap_values: np.ndarray,
                           vital_signs: List[str],
                           sample_idx: int = 0,
                           save_path: str = None):
        if len(shap_values.shape) < 3:
            return
        
        n_features = len(vital_signs)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, vs_name in enumerate(vital_signs):
            if i >= n_features:
                break
            
            temporal_shap = shap_values[sample_idx, :, i]
            axes[i].plot(temporal_shap, linewidth=2)
            axes[i].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('SHAP Value')
            axes[i].set_title(f'{vs_name} Temporal SHAP')
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_feature_importance(self, shap_values: np.ndarray,
                              feature_names: List[str]) -> pd.DataFrame:
        if len(shap_values.shape) == 3:
            shap_abs = np.abs(shap_values).mean(axis=(0, 1))
        elif len(shap_values.shape) == 2:
            shap_abs = np.abs(shap_values).mean(axis=0)
        else:
            shap_abs = np.abs(shap_values)
        
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(shap_abs)],
            'Importance': shap_abs
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def get_temporal_importance(self, shap_values: np.ndarray,
                               vital_signs: List[str]) -> pd.DataFrame:
        if len(shap_values.shape) < 3:
            return pd.DataFrame()
        
        n_samples, seq_len, n_features = shap_values.shape
        temporal_importance = []
        
        for i, vs_name in enumerate(vital_signs):
            if i >= n_features:
                break
            
            vs_shap = shap_values[:, :, i]
            for t in range(seq_len):
                temporal_importance.append({
                    'Vital_Sign': vs_name,
                    'Time_Step': t,
                    'Importance': np.abs(vs_shap[:, t]).mean()
                })
        
        return pd.DataFrame(temporal_importance)
    
    def explain_prediction(self, X_sample: np.ndarray,
                          background_data: np.ndarray,
                          vital_signs: List[str],
                          output_dir: str = 'outputs/shap') -> Dict:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.model_type == 'pytorch':
            results = self.explain_pytorch_model(X_sample, background_data)
        elif self.model_type == 'xgboost':
            results = self.explain_xgboost_model(X_sample, background_data)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        shap_values = results['shap_values']
        
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        
        feature_names = []
        for vs in vital_signs:
            for hour in range(X_sample.shape[1]):
                feature_names.append(f'{vs}_t-{hour}')
        
        importance_df = self.get_feature_importance(shap_values, feature_names)
        
        self.plot_shap_summary(
            shap_values,
            feature_names,
            save_path=str(output_path / 'shap_summary.png')
        )
        
        self.plot_shap_waterfall(
            shap_values,
            results['base_values'],
            feature_names,
            save_path=str(output_path / 'shap_waterfall.png')
        )
        
        if len(shap_values.shape) == 3:
            self.plot_temporal_shap(
                shap_values,
                vital_signs,
                save_path=str(output_path / 'temporal_shap.png')
            )
            
            temporal_importance = self.get_temporal_importance(shap_values, vital_signs)
            if not temporal_importance.empty:
                temporal_importance.to_csv(output_path / 'temporal_importance.csv', index=False)
        
        importance_df.to_csv(output_path / 'feature_importance.csv', index=False)
        
        return {
            'shap_values': shap_values,
            'feature_importance': importance_df,
            'base_value': results['base_values']
        }
