"""
visualization.py

Generates visualizations comparing predicted vs actual weather values.

Author: Krishna Pansari
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

def plot_forecast(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Weather Forecast vs Actual", output_path: Optional[str] = None):
    """
    Plot actual vs predicted values.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        title: Plot title
        output_path: If given, saves the plot to this path
    """
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label='Actual', color='dodgerblue', linewidth=2)
    plt.plot(y_pred, label='Predicted', color='orange', linestyle='--', linewidth=2)
    plt.xlabel("Time Steps")
    plt.ylabel("Forecasted Value")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()
