import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib


sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.rf_model import RandomForestAQI
from src.models.ridge_model import RidgeAQI
from src.models.linear_model import LinearAQI
from src.utils.metrics import evaluate


# Paths
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_data():
    """Load processed data"""
    csv_file = DATA_DIR / "processed_aqi.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Missing processed file: {csv_file}. Run compute_features.py first.")
    
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df):,} rows from {csv_file.name}")
    return df


def prepare_data(df, shuffle=True, test_size=0.2, random_state=42):
    """Split features and target with optional shuffling
    
    Args:
        df: DataFrame with features and target
        shuffle: If True, shuffle data before splitting (use False for time-series)
        test_size: Proportion of data for testing (default 0.2)
        random_state: Random seed for reproducibility
    """
    feature_cols = [
        "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
        "sulphur_dioxide", "ozone", "temperature_2m",
        "relative_humidity_2m", "surface_pressure", "wind_speed_10m",
        "wind_direction_10m", "precipitation"
    ]
    
    X = df[feature_cols].values
    y = df["AQI"].values

    if shuffle:
       
        np.random.seed(random_state)
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        print("✓ Data shuffled before splitting")
    else:
        print("⚠ Using sequential split (no shuffling) - suitable for time-series data")

    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Features: {len(feature_cols)} | Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test, feature_cols


def train_models(X_train, X_test, y_train, y_test, feature_cols):
    """Train Random Forest, Ridge, and Linear models, return best with full metrics"""
    print("\n" + "="*60)
    print("Training Random Forest...")
    print("="*60)
    rf = RandomForestAQI(n_estimators=100, max_depth=20)
    rf.fit(X_train, y_train)
    rf_train = evaluate(y_train, rf.predict(X_train))
    rf_test = evaluate(y_test, rf.predict(X_test))
    importances = rf.feature_importance(feature_cols, top_n=5)
    print("\nTop 5 Important Features:")
    for feature, importance in importances.items():
        print(f"  {feature}: {importance:.4f}")

    print("\n" + "="*60)
    print("Training Ridge Regression...")
    print("="*60)
    ridge = RidgeAQI(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_train = evaluate(y_train, ridge.predict(X_train))
    ridge_test = evaluate(y_test, ridge.predict(X_test))

    print("\n" + "="*60)
    print("Training Linear Regression...")
    print("="*60)
    linear = LinearAQI()
    linear.fit(X_train, y_train)
    linear_train = evaluate(y_train, linear.predict(X_train))
    linear_test = evaluate(y_test, linear.predict(X_test))

    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print("\n  Training Set Metrics:")
    print(f"{'Model':<15} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print("-" * 50)
    print(f"{'Random Forest':<15} {rf_train['rmse']:>10.2f} {rf_train['mae']:>10.2f} {rf_train['r2']:>10.4f}")
    print(f"{'Ridge':<15} {ridge_train['rmse']:>10.2f} {ridge_train['mae']:>10.2f} {ridge_train['r2']:>10.4f}")
    print(f"{'Linear':<15} {linear_train['rmse']:>10.2f} {linear_train['mae']:>10.2f} {linear_train['r2']:>10.4f}")
    
    print("\n Test Set Metrics:")
    print(f"{'Model':<15} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print("-" * 50)
    print(f"{'Random Forest':<15} {rf_test['rmse']:>10.2f} {rf_test['mae']:>10.2f} {rf_test['r2']:>10.4f}")
    print(f"{'Ridge':<15} {ridge_test['rmse']:>10.2f} {ridge_test['mae']:>10.2f} {ridge_test['r2']:>10.4f}")
    print(f"{'Linear':<15} {linear_test['rmse']:>10.2f} {linear_test['mae']:>10.2f} {linear_test['r2']:>10.4f}")

    models = [
        (rf, "random_forest", rf_test),
        (ridge, "ridge", ridge_test),
        (linear, "linear", linear_test)
    ]
    best_model, name, metrics = min(models, key=lambda x: x[2]['rmse'])
    print(f"\n Best Model: {name.replace('_', ' ').title()} (Test RMSE: {metrics['rmse']:.2f})")
    
    return best_model, name, metrics


def save_model(model, name, metrics):
    """Save best model and metrics"""
    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(model, model_path)
    
    metrics_path = MODELS_DIR / "metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"Model: {name}\n")
        f.write(f"RMSE: {metrics['rmse']:.2f}\n")
        f.write(f"MAE:  {metrics['mae']:.2f}\n")
        f.write(f"R²:   {metrics['r2']:.4f}\n")
    
    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")


def main():
    print("="*60)
    print("=== AQI FORECASTING: MODEL TRAINING ===")
    print("="*60)
    
    df = load_data()
    
    # Use shuffle=True for random split, False for time-series sequential split
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(df, shuffle=True)
    
    best_model, name, metrics = train_models(X_train, X_test, y_train, y_test, feature_cols)
    save_model(best_model, name, metrics)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Model: {name}")
    print(f"Test Metrics - RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
