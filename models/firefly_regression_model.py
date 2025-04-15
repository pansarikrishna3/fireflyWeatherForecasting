"""
firefly_regression_model.py

Defines a simple regression model whose weights are optimized using the Firefly Algorithm.
Used for weather forecasting: e.g., predict temperature based on past observations.

Author: Krishna Pansari
"""

import numpy as np
from typing import Tuple
from core.firefly_algorithm import firefly_algorithm


class FireflyRegressor:
    def __init__(self, degree: int = 2):
        self.degree = degree
        self.weights = None

    def _design_matrix(self, X: np.ndarray) -> np.ndarray:
        # Polynomial features: [1, X, X^2, ..., X^d]
        features = [X ** i for i in range(self.degree + 1)]
        return np.hstack(features)

    def _mse(self, weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        X_poly = self._design_matrix(X)
        y_pred = X_poly @ weights
        return np.mean((y - y_pred) ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray, bounds: Tuple[float, float],
            num_fireflies: int = 20, max_generations: int = 50,
            alpha: float = 0.25, beta0: float = 1.0, gamma: float = 1.0,
            verbose: bool = False) -> None:
        dim = self.degree + 1

        def fitness_fn(weights):
            return self._mse(weights, X, y)

        best_weights, _ = firefly_algorithm(
            fitness_fn=fitness_fn,
            dim=dim,
            bounds=bounds,
            num_fireflies=num_fireflies,
            max_generations=max_generations,
            alpha=alpha,
            beta0=beta0,
            gamma=gamma,
            verbose=verbose
        )
        self.weights = best_weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model has not been trained yet.")
        X_poly = self._design_matrix(X)
        return X_poly @ self.weights
