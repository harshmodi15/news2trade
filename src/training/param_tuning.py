import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Tuple
from src.training.trainer import ModelTrainer
from src.utils.config import load_config

class HyperparameterTuner:
    def __init__(self, config_path: str = '../config/hyperparameter_config.yaml'):
        self.config = load_config(config_path=config_path)
        self.search_space = self.config['search_space']
        self.random_search = self.config['random_search']

    def param_tuning(self,
                        train_X: np.ndarray, 
                        train_y: np.ndarray,
                        test_X: np.ndarray,
                        test_y: np.ndarray,
                        scaler,
                        stock_name,
                        scenario: str = "w_Sentiment_and_Keywords",
                        save_dir: str = "../pretrained_model") -> Tuple[pd.DataFrame, Dict]:
        save_dir = Path(f"{save_dir}/{stock_name}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        n_trials = self.random_search['n_trials']
        n_splits = self.random_search['cv_folds']
        
        print(f"Starting {n_trials} trials with {n_splits}-fold cross validation")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        trainer = ModelTrainer()
        ## how to test different window size, how this trial and fold works 
        for trial in range(n_trials):
            print(f"\nTrial {trial + 1}/{n_trials}")
            
            # Sample parameters
            params = self._sample_parameters()
            
            fold_results = []
            for fold, (train_idx, val_idx) in enumerate(tscv.split(train_X), 1):
                print(f"  Fold {fold}/{n_splits}")
                
                # Split data for this fold
                X_train_fold = train_X[train_idx]
                X_val_fold = train_X[val_idx]
                y_train_fold = train_y[train_idx]
                y_val_fold = train_y[val_idx]
                
                try:
                    # Train and evaluate model
                    _, _, metrics = trainer.train_model(
                        train_X=X_train_fold,
                        train_y=y_train_fold,
                        test_X=X_val_fold,
                        test_y=y_val_fold,
                        scaler=scaler,
                        stock_name=stock_name,
                        scenario=scenario,
                        verbose=0
                    )
                    
                    fold_results.append({
                        'fold': fold,
                        **metrics
                    })
                
                except Exception as e:
                    print(f"Error in fold {fold}: {str(e)}")
                    continue
            
            # Calculate average metrics across folds
            if fold_results:
                avg_metrics = self._calculate_average_metrics(fold_results)
                
                result = {
                    **params,
                    **avg_metrics,
                    'n_successful_folds': len(fold_results)
                }
                all_results.append(result)

        # Create results DataFrame and save
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('mean_rmse')
        
        best_params = results_df.iloc[0].to_dict()
        best_model_params = self.get_best_params(results_df)
        # Save results
        results_file = save_dir / f'{stock_name}_{scenario}_cv_results.csv'
        params_file = save_dir / f'{stock_name}_{scenario}_best_params.csv'
        
        results_df.to_csv(results_file, index=False)
        pd.DataFrame([best_model_params]).to_csv(params_file, index=False)
        
        print("\nBest parameters found:")
        for param, value in best_params.items():
            print(f"{param}: {value}")

        return results_df, best_model_params

    def _sample_parameters(self) -> Dict:
        params = {}
        
        for category in ['model', 'training']:
            for param, config in self.search_space[category].items():
                if config['sampling'] == 'choice':
                    params[param] = random.choice(config['values'])
        
        return params

    def _calculate_average_metrics(self, fold_results: List[Dict]) -> Dict:
        avg_metrics = {}
        metrics = [k for k in fold_results[0].keys() if k != 'fold']
        
        for metric in metrics:
            values = [r[metric] for r in fold_results]
            avg_metrics[f'mean_{metric}'] = np.mean(values)
            avg_metrics[f'std_{metric}'] = np.std(values)
        
        return avg_metrics
    
    def get_best_params(self, results_df: pd.DataFrame) -> Dict:
        best_params = results_df.iloc[0].to_dict()
        
        # Initialize dictionary for best parameters
        best_model_params = {}
        
        # Get model parameters
        for param in self.search_space['model'].keys():
            if param in best_params:
                best_model_params[param] = best_params[param]
        
        # Get training parameters
        for param in self.search_space['training'].keys():
            if param in best_params:
                best_model_params[param] = best_params[param]
        
        return best_model_params