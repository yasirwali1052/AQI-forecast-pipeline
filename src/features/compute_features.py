import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _latest_raw_csv() -> Path:
    candidates = sorted(RAW_DIR.glob("openmeteo_combined_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No raw CSVs found in {RAW_DIR}")
    return candidates[-1]


# --- Breakpoint tables ---
PM25_BPS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

PM10_BPS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]

O3_BPS = [
    # 8-hour average (ppb) -> convert µg/m³ to ppb if needed
    (0, 54, 0, 50),
    (55, 70, 51, 100),
    (71, 85, 101, 150),
    (86, 105, 151, 200),
    (106, 200, 201, 300),
]

def _aqi_vectorized(values: pd.Series, bps: list) -> pd.Series:
    """Vectorized AQI calculation for PM2.5, PM10, or O3"""
    aqi = pd.Series(np.nan, index=values.index)
    for cl, ch, il, ih in bps:
        mask = (values >= cl) & (values <= ch)
        aqi.loc[mask] = ((ih - il) / (ch - cl)) * (values.loc[mask] - cl) + il
    return aqi


def _compute_aqi(df: pd.DataFrame) -> pd.Series:
    """Compute AQI vectorized for PM2.5, PM10, O3"""
    aqi_pm25 = _aqi_vectorized(df["pm2_5"], PM25_BPS) if "pm2_5" in df.columns else pd.Series(np.nan, index=df.index)
    aqi_pm10 = _aqi_vectorized(df["pm10"], PM10_BPS) if "pm10" in df.columns else pd.Series(np.nan, index=df.index)
    aqi_o3 = _aqi_vectorized(df["ozone"], O3_BPS) if "ozone" in df.columns else pd.Series(np.nan, index=df.index)
    return pd.concat([aqi_pm25, aqi_pm10, aqi_o3], axis=1).max(axis=1)


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Time features
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    # Rolling means
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for window in (6, 24):
        roll = df[numeric_cols].rolling(window=window, min_periods=max(1, window // 2)).mean()
        roll.columns = [f"{c}_ma{window}" for c in roll.columns]
        df = pd.concat([df, roll], axis=1)

    # Lag features
    for col in ["pm2_5", "pm10", "ozone", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide"]:
        if col in df.columns:
            for lag in (1, 6, 24):
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

    return df


def _build_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    target = df["AQI"]

    drop_cols = {"timestamp", "city", "latitude", "longitude"}
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Drop target leakage
    feature_df = feature_df.drop(columns=["AQI"], errors="ignore")

    # Drop rows with NaNs
    mask = target.notna() & feature_df.notna().all(axis=1)
    return feature_df.loc[mask], target.loc[mask]


def main():
    raw_path = _latest_raw_csv()
    df = pd.read_csv(raw_path)

    # Compute AQI (vectorized)
    df["AQI"] = _compute_aqi(df)

    # Feature engineering
    df_feat = _engineer_features(df)

    # Build X and y
    X, y = _build_xy(df_feat)

    # Save
    stem = raw_path.stem.replace("openmeteo_combined_", "openmeteo_features_")
    features_csv = PROCESSED_DIR / f"{stem}.csv"
    X_csv = PROCESSED_DIR / f"{stem}_X.csv"
    y_csv = PROCESSED_DIR / f"{stem}_y.csv"

    out_df = X.copy()
    out_df["AQI"] = y

    out_df.to_csv(features_csv, index=False)
    X.to_csv(X_csv, index=False)
    y.to_frame(name="AQI").to_csv(y_csv, index=False)

    print("Saved:")
    print(f"  Features: {features_csv}")
    print(f"  X:        {X_csv}")
    print(f"  y:        {y_csv}")


if __name__ == "__main__":
    main()
