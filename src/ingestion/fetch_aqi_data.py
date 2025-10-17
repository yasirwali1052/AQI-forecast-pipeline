
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
from dotenv import load_dotenv

# ==================== CONFIG ====================
load_dotenv()

LAT = float(os.getenv("LAT", "33.6844"))
LON = float(os.getenv("LON", "73.0479"))
CITY = os.getenv("CITY", "Islamabad")
START_DATE = os.getenv("START_DATE")
END_DATE = os.getenv("END_DATE")

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== CORRECTED API ENDPOINTS ====================
# Air Quality (historical)
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Weather ARCHIVE (historical)
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

# Parameters
AIR_PARAMS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone"
]

WEATHER_PARAMS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation"
]


# ==================== HELPERS ====================
def compute_dates():
    """Calculate date range (last 90 days if not specified)"""
    if START_DATE and END_DATE:
        return START_DATE, END_DATE

    end = datetime.utcnow().date()
    start = end - timedelta(days=90)
    return start.isoformat(), end.isoformat()


def fetch_api(url: str, params: Dict) -> Dict:
    """Fetch data from API with error handling"""
    try:
        print(f"Fetching from: {url.split('/')[2]}...")
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        raise


def flatten_hourly(payload: Dict, params: List[str]) -> pd.DataFrame:
    """Convert API response to DataFrame"""
    hourly = payload.get("hourly", {})

    if not hourly.get("time"):
        print("No hourly data in response.")
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    df["timestamp"] = pd.to_datetime(df["time"])
    df.drop(columns=["time"], inplace=True)

    return df


def validate_data(df: pd.DataFrame, name: str) -> bool:
    """Check if DataFrame has valid data"""
    if df.empty:
        print(f"{name} DataFrame is empty.")
        return False

    # Check for missing values
    missing_pct = (df.isnull().sum() / len(df) * 100)

    print(f"\n{name} Data Quality:")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}")

    for col in df.columns:
        if col != 'timestamp' and missing_pct[col] > 0:
            print(f"   {col}: {missing_pct[col]:.1f}% missing")

    if missing_pct.mean() > 50:
        print(f"Warning: {name} has more than 50% missing data overall.")
        return False

    return True


# ==================== MAIN LOGIC ====================
def main():
    """Main execution function"""
    start, end = compute_dates()

    print("Open-Meteo Historical Data Fetcher (FIXED VERSION)")
    print("=" * 60)
    print(f"Location: {CITY} ({LAT}, {LON})")
    print(f"Date Range: {start} → {end}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 60)

    try:
        # ==================== FETCH AIR QUALITY ====================
        print("\nFetching Air Quality Data...")
        air_payload = fetch_api(AIR_URL, {
            "latitude": LAT,
            "longitude": LON,
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(AIR_PARAMS),
            "timezone": "UTC"
        })
        air_df = flatten_hourly(air_payload, AIR_PARAMS)

        if not validate_data(air_df, "Air Quality"):
            print("Air quality data validation failed.")
            return

        print(f"Air Quality: {len(air_df)} hours fetched")

        # ==================== FETCH WEATHER (HISTORICAL) ====================
        print("\nFetching Historical Weather Data...")
        weather_payload = fetch_api(WEATHER_URL, {
            "latitude": LAT,
            "longitude": LON,
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(WEATHER_PARAMS),
            "timezone": "UTC"
        })
        weather_df = flatten_hourly(weather_payload, WEATHER_PARAMS)

        if not validate_data(weather_df, "Weather"):
            print("Weather data validation failed.")
            return

        print(f"Weather: {len(weather_df)} hours fetched")

        # ==================== MERGE BOTH ====================
        print("\nMerging Air Quality + Weather...")

        print(f"   Air Quality timestamps: {air_df['timestamp'].min()} to {air_df['timestamp'].max()}")
        print(f"   Weather timestamps: {weather_df['timestamp'].min()} to {weather_df['timestamp'].max()}")

        merged_df = pd.merge(air_df, weather_df, on="timestamp", how="inner")

        if merged_df.empty:
            print("Merge failed - no overlapping timestamps.")
            return

        print(f"Merged: {len(merged_df)} rows")

        # Add metadata
        merged_df["city"] = CITY
        merged_df["latitude"] = LAT
        merged_df["longitude"] = LON

        # ==================== SAVE FILE ====================
        filename = OUTPUT_DIR / f"openmeteo_combined_{CITY.lower()}_{start.replace('-', '')}-{end.replace('-', '')}.csv"
        merged_df.to_csv(filename, index=False)

        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"File: {filename}")
        print(f"Rows: {len(merged_df)}")
        print(f"Columns: {len(merged_df.columns)}")
        print(f"Date Range: {merged_df['timestamp'].min()} to {merged_df['timestamp'].max()}")

        # ==================== DATA SUMMARY ====================
        print("\nData Summary:")
        print("\nAir Quality Averages:")
        for param in AIR_PARAMS:
            if param in merged_df.columns:
                avg = merged_df[param].mean()
                print(f"   {param:20s}: {avg:8.2f}")

        print("\nWeather Averages:")
        for param in WEATHER_PARAMS:
            if param in merged_df.columns:
                avg = merged_df[param].mean()
                print(f"   {param:30s}: {avg:8.2f}")

        print("\nNext Step: Move to Day 2 - Feature Engineering!")
        print("Run: python src/features/compute_features.py")

    except requests.RequestException as e:
        print(f"\nAPI Error: {e}")
        print("\nTroubleshooting:")
        print("   1. Check internet connection")
        print("   2. Verify coordinates are correct")
        print("   3. Try reducing date range (use last 30 days)")
    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
