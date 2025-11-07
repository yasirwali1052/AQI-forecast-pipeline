"""
Automated Model Retraining Script
Retrains the best model on latest data from feature store
Runs every 24 hours via scheduler
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.feature_store.parquet_store import ParquetFeatureStore
from src.models.rf_model import RandomForestAQI
from src.models.ridge_model import RidgeAQI
from src.models.linear_model import LinearAQI
from src.utils.metrics import evaluate

# ==================== CONFIG ====================
FEATURE_STORE_ROOT = PROJECT_ROOT / "data" / "feature_store" / "parquet"
TABLE_NAME = "aqi_features"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Minimum rows required for retraining
MIN_ROWS_REQUIRED = 100


# ==================== HELPERS ====================
def load_latest_data():
    """Load all data from feature store"""
    try:
        store = ParquetFeatureStore(str(FEATURE_STORE_ROOT))
        df = store.read_table(TABLE_NAME)
        
        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)
        
        print(f"Loaded {len(df):,} rows from feature store")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
        
    except FileNotFoundError:
        print("Feature store not found. Run data ingestion first.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def prepare_training_data(df, test_size=0.2, use_recent_only=False, recent_days=30):
    """Prepare features and target for training
    
    Args:
        df: DataFrame with features and target
        test_size: Proportion for test set
        use_recent_only: If True, use only recent N days for training
        recent_days: Number of recent days to use if use_recent_only=True
    """
    feature_cols = [
        "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
        "sulphur_dioxide", "ozone", "temperature_2m",
        "relative_humidity_2m", "surface_pressure", "wind_speed_10m",
        "wind_direction_10m", "precipitation"
    ]
    
    # Optionally filter to recent data only
    if use_recent_only:
        cutoff_date = df["timestamp"].max() - pd.Timedelta(days=recent_days)
        df = df[df["timestamp"] >= cutoff_date].copy()
        print(f"Using only data from last {recent_days} days: {len(df):,} rows")
    
    # Check for minimum data requirement
    if len(df) < MIN_ROWS_REQUIRED:
        print(f"Insufficient data: {len(df)} rows (minimum {MIN_ROWS_REQUIRED} required)")
        return None, None, None, None, None
    
    # Drop rows with missing values
    df_clean = df.dropna(subset=feature_cols + ["AQI"])
    
    X = df_clean[feature_cols].values
    y = df_clean["AQI"].values
    
    # Time-based split (no shuffling for production)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Features: {len(feature_cols)} | Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    return X_train, X_test, y_train, y_test, feature_cols


def train_and_select_best(X_train, X_test, y_train, y_test, feature_cols):
    """Train multiple models and select the best one"""
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    models = []
    
    # Train Random Forest
    try:
        print("\n[1/3] Training Random Forest...")
        rf = RandomForestAQI(n_estimators=100, max_depth=20, random_state=42)
        rf.fit(X_train, y_train)
        rf_train_metrics = evaluate(y_train, rf.predict(X_train))
        rf_test_metrics = evaluate(y_test, rf.predict(X_test))
        models.append(("Random Forest", rf, rf_train_metrics, rf_test_metrics))
        print(f"   Test RMSE: {rf_test_metrics['rmse']:.2f}, R²: {rf_test_metrics['r2']:.4f}")
    except Exception as e:
        print(f"   Random Forest failed: {e}")
    
    # Train Ridge
    try:
        print("\n[2/3] Training Ridge Regression...")
        ridge = RidgeAQI(alpha=1.0, random_state=42)
        ridge.fit(X_train, y_train)
        ridge_train_metrics = evaluate(y_train, ridge.predict(X_train))
        ridge_test_metrics = evaluate(y_test, ridge.predict(X_test))
        models.append(("Ridge", ridge, ridge_train_metrics, ridge_test_metrics))
        print(f"   Test RMSE: {ridge_test_metrics['rmse']:.2f}, R²: {ridge_test_metrics['r2']:.4f}")
    except Exception as e:
        print(f"   Ridge failed: {e}")
    
    # Train Linear
    try:
        print("\n[3/3] Training Linear Regression...")
        linear = LinearAQI()
        linear.fit(X_train, y_train)
        linear_train_metrics = evaluate(y_train, linear.predict(X_train))
        linear_test_metrics = evaluate(y_test, linear.predict(X_test))
        models.append(("Linear", linear, linear_train_metrics, linear_test_metrics))
        print(f"   Test RMSE: {linear_test_metrics['rmse']:.2f}, R²: {linear_test_metrics['r2']:.4f}")
    except Exception as e:
        print(f"   Linear failed: {e}")
    
    if not models:
        print("\nAll models failed to train")
        return None, None, None
    
    # Select best model by test RMSE
    best_name, best_model, train_metrics, test_metrics = min(models, key=lambda x: x[3]['rmse'])
    
    print("\n" + "="*60)
    print("BEST MODEL SELECTED")
    print("="*60)
    print(f"Model: {best_name}")
    print(f"Test RMSE: {test_metrics['rmse']:.2f}")
    print(f"Test MAE:  {test_metrics['mae']:.2f}")
    print(f"Test R²:   {test_metrics['r2']:.4f}")
    
    return best_model, best_name, test_metrics


def save_model_and_metrics(model, model_name, metrics):
    """Save trained model and metrics"""
    try:
        # Save model
        model_path = MODELS_DIR / "best_model.pkl"
        joblib.dump(model, model_path)
        
        # Save metrics with timestamp
        metrics_path = MODELS_DIR / "metrics.txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        with open(metrics_path, 'w') as f:
            f.write(f"Last Retrained: {timestamp}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Test RMSE: {metrics['rmse']:.2f}\n")
            f.write(f"Test MAE:  {metrics['mae']:.2f}\n")
            f.write(f"Test R²:   {metrics['r2']:.4f}\n")
        
        print(f"\nSaved model to: {model_path}")
        print(f"Saved metrics to: {metrics_path}")
        
        return True
        
    except Exception as e:
        print(f"\nError saving model: {e}")
        return False


# ==================== MAIN ====================
def main():
    """Main retraining pipeline"""
    print("="*60)
    print("AUTOMATED MODEL RETRAINING")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("="*60)
    
    # Load latest data
    df = load_latest_data()
    if df is None or df.empty:
        print("\nRetraining aborted: No data available")
        return False
    
    # Prepare training data (use last 60 days for faster retraining)
    result = prepare_training_data(df, test_size=0.2, use_recent_only=True, recent_days=60)
    
    if result[0] is None:
        print("\nRetraining aborted: Insufficient data")
        return False
    
    X_train, X_test, y_train, y_test, feature_cols = result
    
    # Train models and select best
    best_model, model_name, metrics = train_and_select_best(
        X_train, X_test, y_train, y_test, feature_cols
    )
    
    if best_model is None:
        print("\nRetraining failed: No successful model")
        return False
    
    # Save model and metrics
    success = save_model_and_metrics(best_model, model_name, metrics)
    
    if success:
        print("\n" + "="*60)
        print("RETRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        return True
    else:
        print("\n" + "="*60)
        print("RETRAINING FAILED")
        print("="*60)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)