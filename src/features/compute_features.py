import pandas as pd
import numpy as np
from pathlib import Path

# ==================== CONFIG ====================
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ==================== AQI CALCULATION ====================
# EPA Breakpoints for pollutants (µg/m³ or ppm where applicable)
AQI_BREAKPOINTS = {
    "pm2_5": [(0, 12, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
               (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300)],
    "pm10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
              (255, 354, 151, 200), (355, 424, 201, 300)],
    "ozone": [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150),
               (86, 105, 151, 200)],
    "nitrogen_dioxide": [(0, 53, 0, 50), (54, 100, 51, 100),
                          (101, 360, 101, 150), (361, 649, 151, 200)],
    "sulphur_dioxide": [(0, 35, 0, 50), (36, 75, 51, 100),
                         (76, 185, 101, 150), (186, 304, 151, 200)],
    "carbon_monoxide": [(0, 4.4, 0, 50), (4.5, 9.4, 51, 100),
                         (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200)]
}

AQI_CATEGORIES = {
    (0, 50): "Good",
    (51, 100): "Moderate",
    (101, 150): "Unhealthy for Sensitive Groups",
    (151, 200): "Unhealthy",
    (201, 300): "Very Unhealthy"
}


def compute_aqi_for_pollutant(conc, pollutant):
    """Compute AQI for a given pollutant concentration."""
    for (Clow, Chigh, Ilow, Ihigh) in AQI_BREAKPOINTS.get(pollutant, []):
        if Clow <= conc <= Chigh:
            return ((Ihigh - Ilow) / (Chigh - Clow)) * (conc - Clow) + Ilow
    return np.nan


def compute_overall_aqi(row):
    """Compute overall AQI as the max of individual pollutant AQIs."""
    pollutants = ["pm2_5", "pm10", "ozone", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide"]
    aqi_values = [compute_aqi_for_pollutant(row[p], p) for p in pollutants if not pd.isna(row[p])]
    return max(aqi_values) if aqi_values else np.nan


def categorize_aqi(aqi_value):
    """Assign AQI category."""
    for (low, high), category in AQI_CATEGORIES.items():
        if low <= aqi_value <= high:
            return category
    return "Hazardous"


# ==================== MAIN ====================
def main():
    raw_files = list(RAW_DIR.glob("openmeteo_combined_*.csv"))
    if not raw_files:
        print("No raw data found in data/raw/. Run fetch script first.")
        return

    # Use the most recent file
    latest_file = max(raw_files, key=lambda f: f.stat().st_mtime)
    print(f"Processing file: {latest_file.name}")

    df = pd.read_csv(latest_file)

    # Drop missing pollution values
    df.dropna(subset=["pm2_5", "pm10"], inplace=True)

    # Compute AQI and Category
    df["AQI"] = df.apply(compute_overall_aqi, axis=1)
    df["AQI_Category"] = df["AQI"].apply(categorize_aqi)

    # Drop unnecessary columns (optional)
    df.dropna(subset=["AQI"], inplace=True)

    # Select final features for ML
    feature_cols = [
        "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
        "sulphur_dioxide", "ozone", "temperature_2m",
        "relative_humidity_2m", "surface_pressure", "wind_speed_10m",
        "wind_direction_10m", "precipitation"
    ]
    target_col = "AQI"

    X = df[feature_cols]
    y = df[target_col]

    # Save processed file
    processed_path = PROCESSED_DIR / "processed_aqi.csv"
    df.to_csv(processed_path, index=False)

    print(f" Processed data saved to: {processed_path}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"AQI Range: {df['AQI'].min():.1f} - {df['AQI'].max():.1f}")
    print(f"Categories: {df['AQI_Category'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()
