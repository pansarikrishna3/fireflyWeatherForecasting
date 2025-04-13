"""
main.py

Main script to run weather forecasting using Firefly Algorithm.

Author: Krishna Pansari
"""

import json
import numpy as np
import os
from models.firefly_regression_model import FireflyRegressor
from utils.weather_dataset import load_weather_data, normalize, denormalize
from utils.visualization import plot_forecast

# Load configuration
with open("config/firefly_config.json", "r") as f:
    config = json.load(f)

# Load and prepare data
X, y = load_weather_data("data/weather.csv", feature=config["forecast_feature"], window=config["sliding_window"])
X, x_min, x_max = normalize(X)
y, y_min, y_max = normalize(y)

# Train-test split
split_idx = int(len(X) * config["train_test_split_ratio"])
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train Firefly regression model
model = FireflyRegressor(degree=config["regression_degree"])
model.fit(
    X_train, y_train,
    bounds=(config["lower_bound"], config["upper_bound"]),
    num_fireflies=config["num_fireflies"],
    max_generations=config["max_generations"],
    alpha=config["alpha"],
    beta0=config["beta0"],
    gamma=config["gamma"],
    verbose=True
)

# Predict & evaluate
y_pred = model.predict(X_test)
y_pred_denorm = denormalize(y_pred, y_min, y_max)
y_test_denorm = denormalize(y_test, y_min, y_max)

# Plot results
os.makedirs("outputs", exist_ok=True)
plot_forecast(y_test_denorm, y_pred_denorm, output_path="outputs/forecast_plot.png")

# Save weights & metrics
np.save("outputs/model_weights.npy", model.weights)

metrics = {
    "MSE": float(np.mean((y_test_denorm - y_pred_denorm) ** 2)),
    "MAE": float(np.mean(np.abs(y_test_denorm - y_pred_denorm)))
}
