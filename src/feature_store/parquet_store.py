import os
from typing import Optional

import pandas as pd


class ParquetFeatureStore:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def write_table(self, df: pd.DataFrame, name: str, partition_by: Optional[str] = None) -> str:
        table_dir = os.path.join(self.root_dir, name)
        os.makedirs(table_dir, exist_ok=True)

        if partition_by and partition_by in df.columns:
            df.to_parquet(table_dir, index=False, partition_cols=[partition_by])
        else:
            file_path = os.path.join(table_dir, f"data.parquet")
            df.to_parquet(file_path, index=False)

        return table_dir

    def read_table(self, name: str) -> pd.DataFrame:
        table_dir = os.path.join(self.root_dir, name)
        if not os.path.exists(table_dir):
            raise FileNotFoundError(f"Table not found: {table_dir}")
        return pd.read_parquet(table_dir)


