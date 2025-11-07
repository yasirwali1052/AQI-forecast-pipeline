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


def make_future_feature_frame(reference_df: pd.DataFrame, start_date: datetime, days: int) -> pd.DataFrame:
    """Create synthetic future features with progressive random variation and trends."""
    feature_cols = get_feature_columns()
    rows = []
    
    # Calculate recent trend from last 3 days
    if len(reference_df) >= 3:
        recent_data = reference_df.tail(3)
        trends = {}
        for c in feature_cols:
            # Simple linear trend
            values = recent_data[c].values
            trend = (values[-1] - values[0]) / len(values)
            trends[c] = trend
    else:
        trends = {c: 0 for c in feature_cols}
    
    # Use last day as baseline
    reference_row = reference_df.iloc[-1]
    
    for d in range(days):
        date_value = (start_date + timedelta(days=d)).date()
        row = {"date": date_value}
        for c in feature_cols:
            base = reference_row[c]
            # Apply trend
            trend_effect = trends[c] * (d + 1)
            # Add increasing day-to-day variation (5-15% based on distance)
            variation_range = 0.05 + (d * 0.05)  # Increases variation each day
            drift = base * random.uniform(-variation_range, variation_range)
            row[c] = max(0, base + trend_effect + drift)  # Ensure non-negative
        rows.append(row)
    return pd.DataFrame(rows)


def get_aqi_category(aqi: float) -> tuple:
    """Return category name, color, and health message."""
    if aqi <= 50:
        return "Good", "#00E400", "Air quality is satisfactory"
    elif aqi <= 100:
        return "Moderate", "#FFFF00", "Acceptable for most people"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#FF7E00", "Sensitive groups should limit outdoor activity"
    elif aqi <= 200:
        return "Unhealthy", "#FF0000", "Everyone may experience health effects"
    elif aqi <= 300:
        return "Very Unhealthy", "#8F3F97", "Health alert - everyone at risk"
    else:
        return "Hazardous", "#7E0023", "Health warning - emergency conditions"


def get_pollutant_status(value: float, pollutant: str) -> tuple:
    """Return status and color based on WHO guidelines."""
    thresholds = {
        "pm2_5": [(15, "Good", "#00E400"), (35, "Moderate", "#FFFF00"), (55, "Poor", "#FF7E00"), (float('inf'), "Very Poor", "#FF0000")],
        "pm10": [(45, "Good", "#00E400"), (80, "Moderate", "#FFFF00"), (150, "Poor", "#FF7E00"), (float('inf'), "Very Poor", "#FF0000")],
        "ozone": [(100, "Good", "#00E400"), (140, "Moderate", "#FFFF00"), (180, "Poor", "#FF7E00"), (float('inf'), "Very Poor", "#FF0000")],
    }
    
    if pollutant in thresholds:
        for threshold, status, color in thresholds[pollutant]:
            if value <= threshold:
                return status, color
    return "Unknown", "#CCCCCC"


# ---------------------- STREAMLIT APP ----------------------
def main() -> None:
    st.set_page_config(page_title="AQI Forecast", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .pollutant-card {
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid;
        }
        .insight-box {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #667eea;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("Air Quality Index Forecast")
    st.markdown("**Real-time predictions for the next 3 days** based on historical patterns and environmental factors")
    st.divider()

    with st.spinner("Loading latest data and model..."):
        df = load_feature_store_table()
        model = load_model()

    last7 = get_last_n_days_features(df, n_days=7)
    if last7.empty:
        st.error("No recent data found in the feature store.")
        return

    # Calculate insights
    latest_day = last7.iloc[-1]
    week_ago = last7.iloc[0] if len(last7) > 0 else latest_day
    
    # Main forecast section
    st.subheader("3-Day AQI Forecast")
    st.caption("Predicted Air Quality Index with health implications")
    
    # Predict future 3 days from TODAY (not from last data date)
    feature_cols = get_feature_columns()
    today = datetime.now().date()
    # Always predict starting from tomorrow
    forecast_start = datetime.combine(today, datetime.min.time()) + timedelta(days=1)
    future_df = make_future_feature_frame(last7, forecast_start, days=3)

    X_future = future_df[feature_cols]
    y_pred = model.predict(X_future)
    forecast_df = pd.DataFrame({"date": future_df["date"], "predicted_AQI": y_pred})

    # Forecast cards
    cols = st.columns(3)
    for i, row in forecast_df.iterrows():
        category, color, message = get_aqi_category(row.predicted_AQI)
        
        with cols[i]:
            st.markdown(f"""
                <div style='padding: 25px; border-radius: 12px; background: linear-gradient(135deg, {color}15 0%, {color}35 100%); 
                     border-left: 5px solid {color}; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                    <h4 style='margin: 0; color: #2c3e50; font-weight: 600;'>{row.date.strftime('%a, %b %d')}</h4>
                    <h1 style='margin: 15px 0 10px 0; color: {color}; font-size: 3em; font-weight: bold;'>{row.predicted_AQI:.0f}</h1>
                    <p style='margin: 0; font-size: 15px; font-weight: bold; color: #34495e;'>{category}</p>
                    <p style='margin: 8px 0 0 0; font-size: 13px; color: #7f8c8d; line-height: 1.4;'>{message}</p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two column layout for trends and insights
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Trend Analysis")
        
        # Combine historical and forecast
        historical_aqi = last7.tail(5).copy()
        historical_aqi['AQI'] = historical_aqi['pm2_5'] * 1.5  # Simplified conversion
        historical_aqi['type'] = 'Historical'
        
        forecast_display = forecast_df.copy()
        forecast_display['AQI'] = forecast_display['predicted_AQI']
        forecast_display['type'] = 'Forecast'
        
        combined = pd.concat([
            historical_aqi[['date', 'AQI', 'type']],
            forecast_display[['date', 'AQI', 'type']]
        ])
        combined['date'] = pd.to_datetime(combined['date'])
        
        chart_data = combined.pivot(index='date', columns='type', values='AQI')
        st.line_chart(chart_data, height=350)
    
    with col2:
        st.subheader("Key Pollutants Status")
        st.caption("Current levels compared to WHO guidelines")
        
        # PM2.5 Status
        pm25_status, pm25_color = get_pollutant_status(latest_day['pm2_5'], 'pm2_5')
        st.markdown(f"""
            <div class='pollutant-card' style='border-color: {pm25_color}; background: {pm25_color}15;'>
                <h4 style='margin: 0; color: #2c3e50;'>PM2.5</h4>
                <p style='margin: 5px 0; font-size: 24px; font-weight: bold; color: {pm25_color};'>{latest_day['pm2_5']:.1f} μg/m³</p>
                <p style='margin: 0; font-size: 14px; color: #7f8c8d;'>{pm25_status}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # PM10 Status
        pm10_status, pm10_color = get_pollutant_status(latest_day['pm10'], 'pm10')
        st.markdown(f"""
            <div class='pollutant-card' style='border-color: {pm10_color}; background: {pm10_color}15;'>
                <h4 style='margin: 0; color: #2c3e50;'>PM10</h4>
                <p style='margin: 5px 0; font-size: 24px; font-weight: bold; color: {pm10_color};'>{latest_day['pm10']:.1f} μg/m³</p>
                <p style='margin: 0; font-size: 14px; color: #7f8c8d;'>{pm10_status}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Ozone Status
        o3_status, o3_color = get_pollutant_status(latest_day['ozone'], 'ozone')
        st.markdown(f"""
            <div class='pollutant-card' style='border-color: {o3_color}; background: {o3_color}15;'>
                <h4 style='margin: 0; color: #2c3e50;'>Ozone (O₃)</h4>
                <p style='margin: 5px 0; font-size: 24px; font-weight: bold; color: {o3_color};'>{latest_day['ozone']:.1f} μg/m³</p>
                <p style='margin: 0; font-size: 14px; color: #7f8c8d;'>{o3_status}</p>
            </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Environmental Insights Section
    st.subheader("Environmental Insights")
    
    insight_cols = st.columns(3)
    
    with insight_cols[0]:
        pm25_change = ((latest_day['pm2_5'] - week_ago['pm2_5']) / week_ago['pm2_5'] * 100) if week_ago['pm2_5'] > 0 else 0
        st.markdown(f"""
            <div class='insight-box'>
                <h4 style='color: #667eea; margin: 0 0 10px 0;'>Weekly PM2.5 Change</h4>
                <h2 style='margin: 0; color: {"#e74c3c" if pm25_change > 0 else "#27ae60"};'>{pm25_change:+.1f}%</h2>
                <p style='margin: 10px 0 0 0; color: #7f8c8d; font-size: 14px;'>
                    {"Increasing particulate matter levels" if pm25_change > 0 else "Decreasing particulate matter levels"}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with insight_cols[1]:
        temp_change = latest_day['temperature_2m'] - week_ago['temperature_2m']
        st.markdown(f"""
            <div class='insight-box'>
                <h4 style='color: #667eea; margin: 0 0 10px 0;'>Temperature Trend</h4>
                <h2 style='margin: 0; color: {"#e67e22" if temp_change > 0 else "#3498db"};'>{temp_change:+.1f}°C</h2>
                <p style='margin: 10px 0 0 0; color: #7f8c8d; font-size: 14px;'>
                    Current: {latest_day['temperature_2m']:.1f}°C
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with insight_cols[2]:
        wind_change = ((latest_day['wind_speed_10m'] - week_ago['wind_speed_10m']) / week_ago['wind_speed_10m'] * 100) if week_ago['wind_speed_10m'] > 0 else 0
        st.markdown(f"""
            <div class='insight-box'>
                <h4 style='color: #667eea; margin: 0 0 10px 0;'>Wind Speed Change</h4>
                <h2 style='margin: 0; color: {"#27ae60" if wind_change > 0 else "#e74c3c"};'>{wind_change:+.1f}%</h2>
                <p style='margin: 10px 0 0 0; color: #7f8c8d; font-size: 14px;'>
                    {"Better air circulation" if wind_change > 0 else "Reduced air circulation"}
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Forecast Quality Indicator
    st.divider()
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Health Recommendations")
        avg_forecast_aqi = forecast_df['predicted_AQI'].mean()
        category, _, _ = get_aqi_category(avg_forecast_aqi)
        
        recommendations = {
            "Good": "Great air quality! Perfect for all outdoor activities. Ideal time for exercise and outdoor recreation.",
            "Moderate": "Air quality is acceptable. Unusually sensitive people should consider reducing prolonged outdoor exertion.",
            "Unhealthy for Sensitive Groups": "Sensitive groups (children, elderly, people with respiratory conditions) should reduce prolonged outdoor exertion. General public is less likely to be affected.",
            "Unhealthy": "Everyone should reduce prolonged outdoor exertion. Sensitive groups should avoid outdoor activities. Consider wearing N95 masks outdoors.",
            "Very Unhealthy": "Everyone should avoid prolonged outdoor exertion. Sensitive groups should remain indoors. Use air purifiers and keep windows closed.",
            "Hazardous": "Health emergency! Everyone should avoid all outdoor exertion. Remain indoors, keep windows closed, and use air purifiers."
        }
        
        st.info(f"**Average Forecast Category: {category}**\n\n{recommendations[category]}")
    
    with col2:
        st.subheader("Forecast Details")
        for i, row in forecast_df.iterrows():
            delta = None
            delta_color = "off"
            if i > 0:
                delta = row.predicted_AQI - forecast_df.iloc[i-1].predicted_AQI
                delta_color = "inverse" if delta > 0 else "normal"
            
            st.metric(
                label=f"Day {i+1} - {row.date.strftime('%b %d')}",
                value=f"{row.predicted_AQI:.1f}",
                delta=f"{delta:+.1f} AQI" if delta is not None else "Baseline",
                delta_color=delta_color
            )

    # Historical data table (collapsible)
    with st.expander("View Historical Data (Last 7 Days)"):
        display_df = last7.copy()
        display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
        st.dataframe(display_df.round(2), use_container_width=True, height=300)
    

if __name__ == "__main__":
    main()