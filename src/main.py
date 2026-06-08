import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent))

from data_preprocessing import MIMICDataPreprocessor
from feature_engineering import FeatureEngineer
from training import ModelTrainer
from evaluation import ModelEvaluator
from interpretability import SHAPAnalyzer
from utils import set_seed, setup_logging, save_config


def main():
    parser = argparse.ArgumentParser(description='ICU Vital Sign Forecasting')
    parser.add_argument('--data_path', type=str, default='data/raw',
                       help='Path to MIMIC-III data')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'lstm', 'hybrid', 'transformer', 'tcn', 'xgboost', 'arima'],
                       help='Model to train')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--window_size', type=int, default=24,
                       help='Historical window size (hours)')
    parser.add_argument('--forecast_horizon', type=int, default=6,
                       help='Forecast horizon (hours)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--use_mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--use_lookahead', action='store_true', default=True,
                       help='Use Lookahead optimizer')
    parser.add_argument('--use_tcn', action='store_true', default=True,
                       help='Use TCN in hybrid model')
    
    args = parser.parse_args()
    
    set_seed(42)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir / 'logs')
    
    print("ICU Vital Sign Deterioration Forecasting")
    print("=" * 70)
    
    print("\n[1/6] Data Preprocessing...")
    preprocessor = MIMICDataPreprocessor(
        data_path=args.data_path,
        forecast_horizon=args.forecast_horizon
    )
    
    data = preprocessor.preprocess(
        window_size=args.window_size,
        train_split=0.7,
        val_split=0.15
    )
    
    print("\n[2/6] Feature Engineering...")
    feature_engineer = FeatureEngineer()
    engineered_train, feature_names = feature_engineer.create_engineered_features(data['train']['X'])
    engineered_val, _ = feature_engineer.create_engineered_features(data['val']['X'])
    engineered_test, _ = feature_engineer.create_engineered_features(data['test']['X'])
    print(f"Created {len(feature_names)} engineered features")
    
    data['train']['engineered'] = engineered_train
    data['val']['engineered'] = engineered_val
    data['test']['engineered'] = engineered_test
    data['feature_names'] = feature_names
    
    print(f"\nData Summary:")
    print(f"  Training samples: {data['train']['X'].shape[0]:,}")
    print(f"  Validation samples: {data['val']['X'].shape[0]:,}")
    print(f"  Test samples: {data['test']['X'].shape[0]:,}")
    print(f"  Features: {data['train']['X'].shape[2]}")
    print(f"  Forecast horizon: {args.forecast_horizon} hours")
    
    print("\n[3/6] Model Training...")
    trainer = ModelTrainer(model_dir=output_dir / 'models')
    device = trainer.device
    
    results = {}
    
    if args.model in ['all', 'lstm']:
        print("\nTraining LSTM...")
        lstm_result = trainer.train_lstm(
            data['train'], data['val'],
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=0.001,
            use_mixed_precision=args.use_mixed_precision,
            use_lookahead=args.use_lookahead,
            model_name='lstm',
            hidden_size=256,
            num_layers=3,
            dropout=0.3
        )
        results['LSTM'] = {
            'model': lstm_result['model'],
            'model_type': 'pytorch',
            'history': lstm_result['history']
        }
    
    if args.model in ['all', 'hybrid']:
        print("\nTraining Hybrid LSTM-ARIMA...")
        hybrid_result = trainer.train_hybrid(
            data['train'], data['val'],
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=0.001,
            use_mixed_precision=args.use_mixed_precision,
            use_lookahead=args.use_lookahead,
            model_name='hybrid_lstm_arima',
            hidden_size=256,
            num_layers=3,
            dropout=0.3,
            use_tcn=args.use_tcn
        )
        results['Hybrid LSTM-ARIMA'] = {
            'model': hybrid_result['model'],
            'model_type': 'pytorch',
            'history': hybrid_result['history']
        }
    
    if args.model in ['all', 'transformer']:
        print("\nTraining Transformer...")
        transformer_result = trainer.train_transformer(
            data['train'], data['val'],
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=0.001,
            use_mixed_precision=args.use_mixed_precision,
            use_lookahead=args.use_lookahead,
            model_name='transformer',
            d_model=256,
            nhead=8,
            num_layers=6
        )
        results['Transformer'] = {
            'model': transformer_result['model'],
            'model_type': 'pytorch',
            'history': transformer_result['history']
        }
    
    if args.model in ['all', 'tcn']:
        print("\nTraining TCN...")
        tcn_result = trainer.train_tcn(
            data['train'], data['val'],
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=0.001,
            use_mixed_precision=args.use_mixed_precision,
            use_lookahead=args.use_lookahead,
            model_name='tcn'
        )
        results['TCN'] = {
            'model': tcn_result['model'],
            'model_type': 'pytorch',
            'history': tcn_result['history']
        }
    
    if args.model in ['all', 'xgboost']:
        print("\nTraining XGBoost...")
        xgboost_result = trainer.train_xgboost(
            data['train'], data['val'],
            model_name='xgboost'
        )
        results['XGBoost'] = {
            'model': xgboost_result['model'],
            'model_type': 'xgboost'
        }
    
    if args.model in ['all', 'arima']:
        print("\nTraining ARIMA...")
        arima_result = trainer.train_arima(
            data['train'], data['val'],
            model_name='arima'
        )
        results['ARIMA'] = {
            'model': arima_result['model'],
            'model_type': 'arima'
        }
    
    print("\n[4/6] Model Evaluation...")
    evaluator = ModelEvaluator(norm_params=data['norm_params'])
    
    evaluation_results = {}
    for model_name, model_info in results.items():
        print(f"\nEvaluating {model_name}...")
        eval_result = evaluator.evaluate_model(
            model_info['model'],
            data['test'],
            model_type=model_info['model_type'],
            device=device,
            vital_signs=data['vital_signs'],
            use_uncertainty=False
        )
        evaluation_results[model_name] = eval_result
    
    print("\n[5/6] Model Comparison...")
    comparison_df = evaluator.compare_models(evaluation_results)
    print("\nModel Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    
    if 'per_vital_metrics' in evaluation_results[list(evaluation_results.keys())[0]]:
        per_vital_df = evaluation_results[list(evaluation_results.keys())[0]]['per_vital_metrics']
        per_vital_df.to_csv(output_dir / 'per_vital_metrics.csv', index=False)
        print("\nPer-Vital-Sign Performance:")
        print(per_vital_df.to_string(index=False))
    
    print("\n[6/6] SHAP Interpretability...")
    best_model_name = comparison_df.iloc[0]['Model']
    best_model_info = results[best_model_name]
    
    print(f"\nGenerating SHAP explanations for {best_model_name}...")
    
    shap_analyzer = SHAPAnalyzer(
        best_model_info['model'],
        model_type=best_model_info['model_type'],
        device=device
    )
    
    sample_idx = 0
    X_sample = data['test']['X'][sample_idx:sample_idx+1]
    background_data = data['test']['X'][:100]
    
    shap_results = shap_analyzer.explain_prediction(
        X_sample,
        background_data,
        vital_signs=data['vital_signs'],
        output_dir=str(output_dir / 'shap')
    )
    
    print(f"\nFeature Importance (Top 10):")
    print(shap_results['feature_importance'].head(10).to_string(index=False))
    
    print("\nGenerating visualizations...")
    for model_name, eval_result in evaluation_results.items():
        evaluator.plot_predictions(
            eval_result['true_values'],
            eval_result['predictions'],
            vital_signs=data['vital_signs'],
            save_path=output_dir / f'{model_name.lower().replace(" ", "_")}_predictions.png'
        )
        
        evaluator.plot_forecast_trajectory(
            eval_result['true_values'],
            eval_result['predictions'],
            vital_signs=data['vital_signs'],
            save_path=output_dir / f'{model_name.lower().replace(" ", "_")}_trajectory.png'
        )
    
    config = {
        'data_path': args.data_path,
        'window_size': args.window_size,
        'forecast_horizon': args.forecast_horizon,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'use_mixed_precision': args.use_mixed_precision,
        'use_lookahead': args.use_lookahead,
        'use_tcn': args.use_tcn,
        'best_model': best_model_name,
        'best_rmse': float(comparison_df.iloc[0]['RMSE']),
        'best_mae': float(comparison_df.iloc[0]['MAE']),
        'best_r2': float(comparison_df.iloc[0]['R2'])
    }
    save_config(config, output_dir / 'training_config.json')
    
    print("\n" + "=" * 70)
    print("Training and Evaluation Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print(f"Best model: {best_model_name}")
    print(f"  RMSE: {comparison_df.iloc[0]['RMSE']:.4f}")
    print(f"  MAE: {comparison_df.iloc[0]['MAE']:.4f}")
    print(f"  R²: {comparison_df.iloc[0]['R2']:.4f}")
    if 'Deterioration_F1' in comparison_df.columns:
        print(f"  Deterioration F1: {comparison_df.iloc[0]['Deterioration_F1']:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
