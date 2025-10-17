import pandas as pd

# Load your processed data
df = pd.read_csv("data/processed/openmeteo_features_islamabad_20250718-20251016.csv")

# Find maximum AQI value
max_aqi = df["AQI"].max()

print(f"Maximum AQI value: {max_aqi}")
