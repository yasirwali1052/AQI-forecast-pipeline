"""
Real-time AQI Data Fetcher
Fetches current hourly data from Open-Meteo API and appends to feature store
Runs every 10 minutes (configurable)
"""
import os
import sys
import pandas as pd
import pandas.api.types as ptypes
import requests
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Add project root to PYTHONPATH so absolute imports work when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_store.parquet_store import ParquetFeatureStore
from src.features.compute_features import compute_overall_aqi, categorize_aqi

# ==================== CONFIG ====================
load_dotenv()

LAT = float(os.getenv("LAT", "33.6844"))
LON = float(os.getenv("LON", "73.0479"))
CITY = os.getenv("CITY", "Islamabad")

FEATURE_STORE_ROOT = PROJECT_ROOT / "data" / "feature_store" / "parquet"
TABLE_NAME = "aqi_features"

AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"  # Current weather API

AIR_PARAMS = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
WEATHER_PARAMS = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m", "precipitation"]


# ==================== HELPERS ====================
def _normalize_timestamp(series: pd.Series) -> pd.Series:
    """Return a timezone-naive datetime series tolerant to mixed formats."""
    if ptypes.is_datetime64_any_dtype(series):
        try:
            return series.dt.tz_localize(None)  # type: ignore[return-value]
        except AttributeError:
            return series

    series_str = series.astype(str).str.strip()
    candidates = (
        {"format": "mixed", "utc": True},
        {"format": "ISO8601", "utc": True},
        {"utc": True},
    )

    last_error: Exception | None = None
    for params in candidates:
        try:
            parsed = pd.to_datetime(series_str, errors="raise", **params)
            return parsed.dt.tz_localize(None)
        except (ValueError, TypeError) as error:
            last_error = error
            continue

    cleaned = (
        series_str.str.replace("Z", "", regex=False)
        .str.replace(r"([\+\-]\d{2}:?\d{2})$", "", regex=True)
        .str.replace(r"\.\d{1,9}$", "", regex=True)
    )
    try:
        parsed = pd.to_datetime(cleaned, errors="raise")
        return parsed
    except Exception:
        if last_error is not None:
            raise last_error
        raise


def fetch_current_data():
    """Fetch current hour data from Open-Meteo API"""
    try:
        # Get current time
        now = datetime.utcnow()
        current_date = now.strftime("%Y-%m-%d")
        
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Fetching real-time data for {CITY}...")
        
        # Fetch air quality data (current + past 24h for reliability)
        air_params = {
            "latitude": LAT,
            "longitude": LON,
            "hourly": ",".join(AIR_PARAMS),
            "timezone": "UTC",
            "past_days": 1  # Get last 24h to ensure we have current data
        }
        air_resp = requests.get(AIR_URL, params=air_params, timeout=30)
        air_resp.raise_for_status()
        air_data = air_resp.json().get("hourly", {})
        
        # Fetch weather data
        weather_params = {
            "latitude": LAT,
            "longitude": LON,
            "hourly": ",".join(WEATHER_PARAMS),
            "timezone": "UTC",
            "past_days": 1
        }
        weather_resp = requests.get(WEATHER_URL, params=weather_params, timeout=30)
        weather_resp.raise_for_status()
        weather_data = weather_resp.json().get("hourly", {})
        
        if not air_data or not weather_data:
            print("❌ Empty response from API")
            return None
        
        # Convert to DataFrames
        air_df = pd.DataFrame(air_data)
        weather_df = pd.DataFrame(weather_data)
        
        # Parse timestamps
        air_df["timestamp"] = pd.to_datetime(air_df["time"])
        weather_df["timestamp"] = pd.to_datetime(weather_df["time"])
        
        air_df.drop(columns=["time"], inplace=True)
        weather_df.drop(columns=["time"], inplace=True)
        
        # Merge air and weather
        df = pd.merge(air_df, weather_df, on="timestamp", how="inner")
        
        # Get only the latest complete hour
        df = df.sort_values("timestamp").tail(1)
        
        if df.empty:
            print("❌ No data available for current hour")
            return None
        
        # Add metadata
        df["city"] = CITY
        df["latitude"] = LAT
        df["longitude"] = LON
        
        # Compute AQI
        df["AQI"] = df.apply(compute_overall_aqi, axis=1)
        df["AQI_Category"] = df["AQI"].apply(categorize_aqi)
        
        # Drop rows with missing critical values
        df.dropna(subset=["pm2_5", "pm10", "AQI"], inplace=True)
        
        if df.empty:
            print("❌ No valid data after processing")
            return None
        
        print(f"✅ Fetched {len(df)} row(s) - Latest: {df['timestamp'].iloc[0]}")
        print(f"   AQI: {df['AQI'].iloc[0]:.1f} ({df['AQI_Category'].iloc[0]})")
        print(f"   PM2.5: {df['pm2_5'].iloc[0]:.1f} μg/m³, PM10: {df['pm10'].iloc[0]:.1f} μg/m³")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API Request failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None


def append_to_feature_store(df):
    """Append new data to parquet feature store"""
    try:
        df = df.copy()
        df["timestamp"] = _normalize_timestamp(df["timestamp"])
        store = ParquetFeatureStore(str(FEATURE_STORE_ROOT))
        
        # Read existing data
        try:
            existing_df = store.read_table(TABLE_NAME)
            existing_df["timestamp"] = _normalize_timestamp(existing_df["timestamp"])
            
            # Check for duplicates by timestamp
            new_timestamps = df["timestamp"].values
            mask = ~existing_df["timestamp"].isin(new_timestamps)
            
            if not mask.any():
                print("⚠️  Data already exists for this timestamp - skipping")
                return False
            
            # Append new data
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.sort_values("timestamp", inplace=True)
            combined_df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
            
            print(f"📊 Appending to existing store: {len(existing_df)} + {len(df)} = {len(combined_df)} rows")
            
        except FileNotFoundError:
            print("📦 Creating new feature store")
            combined_df = df
        
        # Add date partition
        combined_df["timestamp"] = _normalize_timestamp(combined_df["timestamp"])
        combined_df["date_partition"] = combined_df["timestamp"].dt.date.astype(str)
        
        # Write back to store
        store.write_table(combined_df, TABLE_NAME, partition_by="date_partition")
        
        print(f"✅ Successfully updated feature store at {FEATURE_STORE_ROOT / TABLE_NAME}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating feature store: {e}")
        return False


# ==================== MAIN ====================
def main():
    """Main execution function"""
    print("="*60)
    print("🌍 REAL-TIME AQI DATA FETCHER")
    print("="*60)
    
    # Fetch current data
    df = fetch_current_data()
    
    if df is None or df.empty:
        print("\n⚠️  No new data fetched - will retry on next run")
        return False
    
    # Append to feature store
    success = append_to_feature_store(df)
    
    if success:
        print("\n✅ Real-time update completed successfully")
        print("="*60)
        return True
    else:
        print("\n❌ Failed to update feature store")
        print("="*60)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)