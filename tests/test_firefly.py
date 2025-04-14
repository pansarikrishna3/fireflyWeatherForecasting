"""
test_firefly.py

Unit tests for the Firefly Algorithm core logic.

Author: Krishna Pansari
"""

import unittest
import numpy as np
from core.firefly_algorithm import firefly_algorithm


class TestFireflyAlgorithm(unittest.TestCase):

    def setUp(self):
        self.dim = 3
        self.bounds = (-5.0, 5.0)
        self.fitness_fn = lambda x: np.sum(x ** 2)  # Global minimum at 0

    def test_output_shape_and_type(self):
        best_solution, best_score = firefly_algorithm(
            fitness_fn=self.fitness_fn,
            dim=self.dim,
            bounds=self.bounds,
            num_fireflies=10,
            max_generations=5
        )
        self.assertIsInstance(best_solution, np.ndarray)
        self.assertEqual(best_solution.shape[0], self.dim)
        self.assertIsInstance(best_score, float)

    def test_solution_within_bounds(self):
        best_solution, _ = firefly_algorithm(
            fitness_fn=self.fitness_fn,
            dim=self.dim,
            bounds=self.bounds,
            num_fireflies=10,
            max_generations=5
        )
        self.assertTrue(np.all(best_solution >= self.bounds[0]))
        self.assertTrue(np.all(best_solution <= self.bounds[1]))

    def test_decreasing_fitness(self):
        # Optional: test fitness improves or at least doesn't worsen drastically
        initial = self.fitness_fn(np.array([2.0, 2.0, 2.0]))
        _, final = firefly_algorithm(
            fitness_fn=self.fitness_fn,
            dim=self.dim,
            bounds=self.bounds,
            num_fireflies=15,
            max_generations=20
        )
        self.assertLessEqual(final, initial)


if __name__ == '__main__':
    unittest.main()
