"""
test_weather_dataset.py

Unit tests for weather_dataset.py: data loading, normalization, and window slicing.

Author: Krishna Pansari
"""

import unittest
import numpy as np
import pandas as pd
import os
from utils.weather_dataset import load_weather_data, normalize, denormalize

class TestWeatherDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_csv = "temp_weather.csv"
        df = pd.DataFrame({
            "temperature": np.linspace(10, 20, 15)
        })
        df.to_csv(cls.temp_csv, index=False)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.temp_csv)

    def test_load_weather_data_shapes(self):
        X, y = load_weather_data(self.temp_csv, feature="temperature", window=3)
        self.assertEqual(X.shape[1], 3)
        self.assertEqual(X.shape[0], y.shape[0])

    def test_normalization_and_denormalization(self):
        arr = np.array([10.0, 20.0, 15.0, 25.0])
        norm, min_v, max_v = normalize(arr)
        self.assertTrue(np.all((0.0 <= norm) & (norm <= 1.0)))
        original = denormalize(norm, min_v, max_v)
        np.testing.assert_allclose(original, arr, rtol=1e-5)

    def test_missing_feature_error(self):
        with self.assertRaises(ValueError):
            load_weather_data(self.temp_csv, feature="humidity", window=3)


if __name__ == "__main__":
    unittest.main()

