import pandas as pd
import numpy as np

np.random.seed(42)
days = 100
base_temp = 20 + 5 * np.sin(np.linspace(0, 4 * np.pi, days))  # seasonal pattern
noise = np.random.normal(0, 1, days)
temperature = base_temp + noise

df = pd.DataFrame({
    "day": np.arange(1, days + 1),
    "temperature": temperature.round(2)
})

df.to_csv("data/weather.csv", index=False)
print("✅ weather.csv created in /data/")
