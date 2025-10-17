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


def _aqi_from_pm25(pm25: float) -> float:
    # EPA breakpoint table for PM2.5 (µg/m³)
    # (Clow, Chigh, Ilow, Ihigh)
    bps = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    if np.isnan(pm25):
        return np.nan
    for cl, ch, il, ih in bps:
        if cl <= pm25 <= ch:
            return (ih - il) / (ch - cl) * (pm25 - cl) + il
    return np.nan


def _aqi_from_pm10(pm10: float) -> float:
    # EPA breakpoint table for PM10 (µg/m³)
    bps = [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500),
    ]
    if np.isnan(pm10):
        return np.nan
    for cl, ch, il, ih in bps:
        if cl <= pm10 <= ch:
            return (ih - il) / (ch - cl) * (pm10 - cl) + il
    return np.nan


def _compute_aqi(row: pd.Series) -> float:
    aqi_pm25 = _aqi_from_pm25(row.get("pm2_5", np.nan))
    aqi_pm10 = _aqi_from_pm10(row.get("pm10", np.nan))
    return np.nanmax([aqi_pm25, aqi_pm10])


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Time features
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    # Rolling means to smooth noisy signals (use past 6 and 24 hours)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for window in (6, 24):
        roll = (
            df[numeric_cols]
            .rolling(window=window, min_periods=max(1, window // 2))
            .mean()
            .add_suffix(f"_ma{window}")
        )
        df = pd.concat([df, roll], axis=1)

    # Lag features for primary pollutants (1, 6, 24 hours)
    for col in ["pm2_5", "pm10", "ozone", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide"]:
        if col in df.columns:
            for lag in (1, 6, 24):
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

    return df


def _build_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    target = df["AQI"]

    drop_cols = {
        "timestamp",
        "city",
        "latitude",
        "longitude",
        # raw pollutant columns can still be useful; keep them
    }
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Remove target leakage by dropping any direct AQI-like columns if present
    feature_df = feature_df.drop(columns=["AQI"], errors="ignore")

    # Drop rows with missing target or empty feature rows
    mask = target.notna()
    feature_df = feature_df.loc[mask]
    target = target.loc[mask]

    # After lags/rolling, initial rows may contain NaNs; drop remaining NaNs
    valid_mask = feature_df.notna().all(axis=1)
    feature_df = feature_df.loc[valid_mask]
    target = target.loc[valid_mask]

    return feature_df, target


def main():
    raw_path = _latest_raw_csv()
    df = pd.read_csv(raw_path)

    # Compute AQI
    df["AQI"] = df.apply(_compute_aqi, axis=1)

    # Feature engineering
    df_feat = _engineer_features(df)

    # Build X and y
    X, y = _build_xy(df_feat)

    # Output filenames (align with test path pattern)
    # Try to mirror the date range suffix from the raw filename if present
    stem = raw_path.stem.replace("openmeteo_combined_", "openmeteo_features_")
    features_csv = PROCESSED_DIR / f"{stem}.csv"
    X_csv = PROCESSED_DIR / f"{stem}_X.csv"
    y_csv = PROCESSED_DIR / f"{stem}_y.csv"

    # Join X and y for a single features file as well
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

