
import os
import json
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Tuple, Dict
from src.models.lstm_models import LSTMModelFactory
from src.training.trainer import ModelTrainer
from src.utils.metrics import calculate_metrics
from src.utils.config import load_config

class ModelEvaluator:
    def __init__(self):
        self.config = load_config()

    def final_training(self,
                        train_X: np.ndarray, 
                        train_y: np.ndarray,
                        test_X: np.ndarray,
                        test_y: np.ndarray,
                        scaler,
                        stock_name: str,
                        scenario: str = "w_Sentiment_and_Keywords",
                        n_runs: int = 1) -> Tuple[object, pd.DataFrame, str]:
        
        save_dir = Path(f"../pretrained_model/{stock_name}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load best parameters
        best_params_path = save_dir / f'{stock_name}_{scenario}_best_params.csv'
        best_params = pd.read_csv(best_params_path).iloc[0].to_dict()
        
        print(f"\nTraining {n_runs} models for {scenario}")
        print("Best parameters:", best_params)
        
        # Track results
        all_metrics = []
        best_model = None
        best_rmse = float('inf')
        best_r2 = float('inf')
        best_run = None
        trainer = ModelTrainer()

        # Train multiple runs
        for run in range(1, n_runs + 1):
            print(f"\nRun {run}/{n_runs}")
            
            try:
                model, _, metrics = trainer.train_model(
                        train_X=train_X,
                        train_y=train_y,
                        test_X=test_X,
                        test_y=test_y,
                        scaler=scaler,
                        stock_name=stock_name,
                        scenario=scenario,
                        verbose=0
                    )

                metrics['run'] = run
                all_metrics.append(metrics)
                
                # Track best model
                if metrics['rmse'] < best_rmse and metrics['r2'] < best_r2:
                    best_rmse = metrics['rmse']
                    best_r2 = metrics['r2']
                    best_run = run
                    best_model = model
                    print(f"\nNew best model! RMSE: {best_rmse:.4f}, R^2: {best_r2:.4f}")
                
            except Exception as e:
                print(f"Error in run {run}: {str(e)}")
                continue
        
        # Save results
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(save_dir / f'{stock_name}_{scenario}_final_metrics.csv', index=False)
        
        if best_model is not None:
            # Save best model
            best_model.save(save_dir / f'{stock_name}_{scenario}_best_model.keras')
            print("Best model Saved")
            # Save configuration
            config = {
                'best_model_info': {
                    'run': best_run,
                    'metrics': metrics_df.loc[metrics_df['run'] == best_run].to_dict('records')[0]
                },
                'training_params': {
                    **best_params,
                    'n_runs': n_runs,
                    'scenario': scenario
                },
                'summary_metrics': {
                    'best_rmse': float(best_rmse),
                    'best_r2': float(best_r2)
                }
            }
            
            with open(save_dir / f'{stock_name}_{scenario}_config.json', 'w') as f:
                json.dump(config, f, indent=4)
        
        # Print summary
        print("\nPerformance Summary:")
        summary = metrics_df.agg(['mean', 'std', 'min', 'max'])
        print(summary)
        
        print(f"\nBest Model - Run {best_run} - RMSE: {best_rmse:.4f} - R^2: {best_r2:.4f}")
        
        return best_model, metrics_df, save_dir