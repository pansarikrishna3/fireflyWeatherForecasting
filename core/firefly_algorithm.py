"""
firefly_algorithm.py

Core implementation of the Firefly Algorithm (FA) for optimizing weather forecasting model parameters.

Author: Krishna Pansari
"""

import numpy as np
import random
from typing import Callable, Tuple


# === Firefly Algorithm ===
def firefly_algorithm(
    fitness_fn: Callable[[np.ndarray], float],
    dim: int,
    bounds: Tuple[float, float],
    num_fireflies: int = 20,
    max_generations: int = 50,
    alpha: float = 0.25,
    beta0: float = 1.0,
    gamma: float = 1.0,
    verbose: bool = False
) -> Tuple[np.ndarray, float]:
    """
    Firefly Algorithm to minimize a fitness function.

    Args:
        fitness_fn: Function to minimize
        dim: Number of dimensions (e.g., number of model weights)
        bounds: Tuple (lower_bound, upper_bound)
        num_fireflies: Population size
        max_generations: Maximum iterations
        alpha: Randomness coefficient
        beta0: Base attractiveness
        gamma: Light absorption coefficient
        verbose: Print fitness progress

    Returns:
        Best solution found and its fitness score
    """
    lower_bound, upper_bound = bounds
    # Initialize population
    fireflies = np.random.uniform(lower_bound, upper_bound, size=(num_fireflies, dim))
    fitness = np.array([fitness_fn(f) for f in fireflies])

    # Main loop
    for gen in range(max_generations):
        for i in range(num_fireflies):
            for j in range(num_fireflies):
                if fitness[j] < fitness[i]:
                    rij = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta = beta0 * np.exp(-gamma * rij**2)
                    # Move firefly i toward j
                    attraction = beta * (fireflies[j] - fireflies[i])
                    randomness = alpha * (np.random.rand(dim) - 0.5)
                    fireflies[i] += attraction + randomness
                    # Boundary check
                    fireflies[i] = np.clip(fireflies[i], lower_bound, upper_bound)
                    fitness[i] = fitness_fn(fireflies[i])
        if verbose:
            print(f"Generation {gen+1:02d}: Best Fitness = {np.min(fitness):.6f}")

    best_idx = np.argmin(fitness)
    return fireflies[best_idx], fitness[best_idx]
