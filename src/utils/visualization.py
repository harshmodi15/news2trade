import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from datetime import datetime

def plot_training_history(ax: plt.Axes, 
                         history: Dict, 
                         title: str) -> None:
    history_dict = history.history
    
    ax.plot(history_dict['loss'], label='Training Loss', linewidth=2, color='#2ecc71')
    if 'val_loss' in history_dict:
        ax.plot(history_dict['val_loss'], label='Validation Loss', linewidth=2, color='#e74c3c')
    
    ax.set_title(title, pad=20, fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(frameon=True)

def plot_price_comparison(ax: plt.Axes, 
                         actual: np.ndarray, 
                         predicted: np.ndarray,
                         dates: Optional[Union[pd.Series, np.ndarray]] = None,
                         title: str = "Stock Price Prediction") -> None:
    ax.fill_between(range(len(actual)),
                   actual,
                   predicted,
                   alpha=0.2,
                   color='gray',
                   label='Difference')
    
    ax.plot(actual, label='Actual', color='#2980b9', marker='o', markersize=2, linewidth=2)
    ax.plot(predicted, label='Predicted', color='#e74c3c', marker='x', markersize=2, linestyle= '--',linewidth=2)

    if dates is not None:
        format_date_axis(ax, dates)

    ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Stock Price', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(frameon=True, fontsize=10, loc='upper left')

def format_date_axis(ax: plt.Axes, dates: Union[pd.Series, np.ndarray]) -> None:
    if isinstance(dates.iloc[0], str):
        dates = pd.to_datetime(dates)
    
    n_ticks = 20
    tick_indices = np.linspace(0, len(dates)-1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([dates.iloc[i].strftime('%Y-%m-%d') for i in tick_indices], 
                       rotation=45)

def plot_results(stock_name: str,
                history: Dict,
                actual: np.ndarray,
                prediction: np.ndarray,
                dates: Optional[Union[pd.Series, np.ndarray]] = None,
                model_type: str = "",
                scenario_name: str = "",
                show_training: bool = True) -> plt.Figure:
    if show_training:
        fig = plt.figure(figsize=(18, 6))
        
        # Training history plot
        ax1 = plt.subplot(121)
        plot_training_history(
            ax1, 
            history, 
            f'Training History\n{stock_name} - {scenario_name} - {model_type}'
        )
        
        # Price prediction plot
        ax2 = plt.subplot(122)
    else:
        fig = plt.figure(figsize=(15, 8))
        ax2 = plt.gca()
    
    plot_price_comparison(
        ax2,
        actual,
        prediction,
        dates,
        f'{stock_name} Stock Price Prediction\n{scenario_name}'
    )
    
    plt.tight_layout()
    return fig

# Helper function for saving plots
def save_plot(fig: plt.Figure, 
             filename: str, 
             directory: str = "results/plots") -> None:
    """Save the plot to a file."""
    import os
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close(fig)