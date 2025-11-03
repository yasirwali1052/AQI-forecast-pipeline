import os
import pandas as pd

try:
    
    from src.feature_store.parquet_store import ParquetFeatureStore
except Exception:
   
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.feature_store.parquet_store import ParquetFeatureStore


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "processed_aqi.csv")
STORE_ROOT = os.path.join(PROJECT_ROOT, "data", "feature_store", "parquet")
TABLE_NAME = "aqi_features"


def main() -> None:
    if not os.path.exists(DATA_CSV_PATH):
        raise FileNotFoundError(f"Missing input CSV: {DATA_CSV_PATH}")

    df = pd.read_csv(DATA_CSV_PATH)

   
    partition_col = None
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower() or "timestamp" in col.lower():
            try:
                ts = pd.to_datetime(df[col], errors="raise")
                df["date_partition"] = ts.dt.date.astype(str)
                partition_col = "date_partition"
                break
            except Exception:
                continue

    store = ParquetFeatureStore(STORE_ROOT)
    output_dir = store.write_table(df, TABLE_NAME, partition_by=partition_col)
    print(f"Wrote Parquet table to: {output_dir}")


if __name__ == "__main__":
    main()