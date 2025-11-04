import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import joblib
import pandas as pd
import streamlit as st

# ---------------------- PATH CONFIG ----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_STORE_ROOT = PROJECT_ROOT / "data" / "feature_store" / "parquet"
FEATURE_TABLE = "aqi_features"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"


# ---------------------- FEATURE LIST ----------------------
def get_feature_columns() -> List[str]:
    return [
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
        "precipitation",
    ]


# ---------------------- LOADERS ----------------------
@st.cache_data(show_spinner=False)
def load_feature_store_table() -> pd.DataFrame:
    try:
        from src.feature_store.parquet_store import ParquetFeatureStore
    except Exception:
        import sys
        sys.path.append(str(PROJECT_ROOT))
        from src.feature_store.parquet_store import ParquetFeatureStore

    store = ParquetFeatureStore(str(FEATURE_STORE_ROOT))
    df = store.read_table(FEATURE_TABLE)
    ts_col_candidates = [c for c in df.columns if c.lower() in {"timestamp", "datetime", "date"}]
    if not ts_col_candidates:
        raise ValueError("No timestamp-like column found (expected one of: timestamp, datetime, date).")
    ts_col = ts_col_candidates[0]
    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df.dropna(subset=["timestamp"], inplace=True)
    df["date"] = df["timestamp"].dt.tz_convert("UTC").dt.date
    return df


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


# ---------------------- FEATURE PROCESSING ----------------------
def get_last_n_days_features(df: pd.DataFrame, n_days: int = 7) -> pd.DataFrame:
    feature_cols = get_feature_columns()
    daily = df.groupby("date", as_index=False)[feature_cols].mean().sort_values("date")
    if daily.empty:
        return daily
    last_date = daily["date"].max()
    start_date = last_date - timedelta(days=n_days - 1)
    mask = (daily["date"] >= start_date) & (daily["date"] <= last_date)
    return daily.loc[mask].reset_index(drop=True)


def make_future_feature_frame(reference_row: pd.Series, start_date: datetime, days: int) -> pd.DataFrame:
    """Create synthetic future features with small random drift."""
    feature_cols = get_feature_columns()
    rows = []
    for d in range(days):
        date_value = (start_date + timedelta(days=d)).date()
        row = {"date": date_value}
        for c in feature_cols:
            base = reference_row[c]
            # Add slight day-to-day variation (+/- up to 7%)
            drift = base * random.uniform(-0.07, 0.07)
            row[c] = base + drift
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------- STREAMLIT APP ----------------------
def main() -> None:
    st.set_page_config(page_title="🌫️ AQI Forecast", layout="centered")
    st.title("🌫️ AQI Forecast Dashboard")
    st.markdown("Predicts the next **3 days** of Air Quality Index (AQI) using the latest feature data.")
    st.divider()

    with st.spinner("🔄 Loading latest data and model..."):
        df = load_feature_store_table()
        model = load_model()

    last7 = get_last_n_days_features(df, n_days=7)
    if last7.empty:
        st.error("⚠️ No recent data found in the feature store.")
        return

    st.subheader("📅 Latest 7 Days (Averaged Features)")
    st.dataframe(last7, use_container_width=True)

    # Predict future 3 days
    feature_cols = get_feature_columns()
    last_date = last7["date"].max()
    forecast_start = datetime.combine(last_date, datetime.min.time()) + timedelta(days=1)
    reference_row = last7[last7["date"] == last_date].iloc[0]
    future_df = make_future_feature_frame(reference_row, forecast_start, days=3)

    X_future = future_df[feature_cols]
    y_pred = model.predict(X_future)
    forecast_df = pd.DataFrame({"date": future_df["date"], "predicted_AQI": y_pred})

    # ---------------------- UI DISPLAY ----------------------
    st.subheader("🔮 Next 3-Day AQI Forecast")
    cols = st.columns(3)
    for i, row in forecast_df.iterrows():
        color = "🟢" if row.predicted_AQI < 100 else "🟠" if row.predicted_AQI < 150 else "🔴"
        cols[i].metric(
            label=f"{color} {row.date}",
            value=f"{row.predicted_AQI:.2f}",
            delta="Predicted AQI"
        )

    st.line_chart(forecast_df.set_index("date")["predicted_AQI"], height=320)
    


if __name__ == "__main__":
    main()
