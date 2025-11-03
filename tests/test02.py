import sqlite3
import pandas as pd

conn = sqlite3.connect(r"E:\AQI-forecast-pipeline\features.db")

# Use the actual table name printed from check_tables.py
count = pd.read_sql_query("SELECT COUNT(*) as total_rows FROM processed_features;", conn)

print(count)

conn.close()
