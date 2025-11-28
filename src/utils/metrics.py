import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from typing import Dict, Union
from tabulate import tabulate
from datetime import timedelta

def calculate_metrics(actual: np.ndarray, prediction: np.ndarray, training_time: float) -> Dict[str, float]:
    epsilon = 1e-8
    actual_safe = np.where(actual == 0, epsilon, actual)
    
    mse = mean_squared_error(actual, prediction)
    rmse = sqrt(mse)
    mae = mean_absolute_error(actual, prediction)
    r2 = r2_score(actual, prediction)
    
    mape = np.mean(np.abs((actual - prediction) / actual_safe)) * 100
    smape = np.mean(np.abs(prediction - actual) / ((np.abs(actual) + np.abs(prediction)) / 2)) * 100
    mpe = np.mean((prediction - actual) / actual_safe) * 100
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape),
        'smape': float(smape),
        'mpe': float(mpe),
        'training_time': training_time
    }

def format_metrics(metrics: Dict[str, float]) -> str:
    headers = ["Metric", "Value"]
    rows = []
    
    time_str = str(timedelta(seconds=int(metrics['training_time'])))
    
    metrics_display = [
        ("Training Time", time_str),
        ("MSE", f"{metrics['mse']:.4f}"),
        ("RMSE", f"{metrics['rmse']:.4f}"),
        ("MAE", f"{metrics['mae']:.4f}"),
        ("R²", f"{metrics['r2']:.4f}"),
        ("MAPE (%)", f"{metrics['mape']:.2f}"),
        ("SMAPE (%)", f"{metrics['smape']:.2f}"),
        ("MPE (%)", f"{metrics['mpe']:.2f}")
    ]
    print(tabulate(metrics_display, headers=headers, tablefmt="grid"))
    return tabulate(metrics_display, headers=headers, tablefmt="grid")

def print_metrics_comparison(models_metrics: Dict[str, Dict[str, float]]) -> str:
    metric_names = ['MSE', 'RMSE', 'MAE', 'R²', 'MAPE (%)', 'SMAPE (%)', 'MPE (%)', 'Training Time']
    headers = ['Model'] + metric_names
    
    rows = []
    for model_name, metrics in models_metrics.items():
        time_str = str(timedelta(seconds=int(metrics['training_time'])))
        row = [
            model_name,
            f"{metrics['mse']:.4f}",
            f"{metrics['rmse']:.4f}",
            f"{metrics['mae']:.4f}",
            f"{metrics['r2']:.4f}",
            f"{metrics['mape']:.2f}",
            f"{metrics['smape']:.2f}",
            f"{metrics['mpe']:.2f}",
            time_str
        ]
        rows.append(row)
    
    return tabulate(rows, headers=headers, tablefmt="grid")