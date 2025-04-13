"""
weather_dataset.py

Loads and preprocesses weather data for forecasting tasks.
Assumes CSV input with time-series features like temperature, humidity, etc.

Author: Krishna Pansari
"""

import pandas as pd
import numpy as np
from typing import Tuple

def load_weather_data(csv_path: str, feature: str = "temperature", window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads weather data and creates input/output pairs for forecasting.

    Args:
        csv_path: Path to CSV file
        feature: Column to forecast (e.g., "temperature")
        window: Number of previous time steps to use as input

    Returns:
        Tuple of (X, y) where:
        - X is a 2D array of shape (n_samples, window)
        - y is a 1D array of shape (n_samples,)
    """
    df = pd.read_csv(csv_path)
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in dataset.")

    series = df[feature].values.astype(float)

    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])

    return np.array(X), np.array(y)

def normalize(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    min_val = np.min(data)
    max_val = np.max(data)
    norm = (data - min_val) / (max_val - min_val + 1e-8)
    return norm, min_val, max_val

def denormalize(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    return data * (max_val - min_val + 1e-8) + min_val
