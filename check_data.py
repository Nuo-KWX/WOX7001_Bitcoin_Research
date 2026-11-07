
import pandas as pd
import numpy as np

# 1. Read the downloaded CSV
file_path = "data/raw/binance/BTCUSDT_Daily_2021_2025.csv"

columns = [
    "open_time", "open", "high", "low", "close",
    "volume", "close_time", "quote_volume", "trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
]

df = pd.read_csv(file_path)

# 2. Check the dataset structure
print("===== DATASET OVERVIEW =====")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# 3. Convert timestamps
timestamp = pd.to_numeric(df["open_time"], errors="coerce")

# Binance uses milliseconds in older files
# and microseconds in newer Spot files.

microsecond_mask = timestamp >= 1e14

# Convert all timestamps into UTC
dates = pd.Series(
    pd.to_datetime(timestamp, unit="ms", utc=True),
    index=df.index
)

# Convert newer microsecond timestamps
dates.loc[microsecond_mask] = pd.to_datetime(
    timestamp.loc[microsecond_mask],
    unit="us",
    utc=True
)

df["date"] = dates

# 4. Convert numerical columns
numeric_columns = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "trades",
    "taker_buy_base_volume", "taker_buy_quote_volume"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 5. Check date range
print("\n===== DATE RANGE =====")
print("Start date:", df["date"].min())
print("End date:", df["date"].max())

# 6. Check missing values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# 7. Check duplicate dates
print("\n===== DUPLICATE DATES =====")
print("Duplicate dates:", df["date"].duplicated().sum())

# 8. Check chronological order
print("\n===== DATE ORDER =====")
print("Chronologically sorted:", df["date"].is_monotonic_increasing)

# 9. Check OHLC and volume validity
invalid_ohlc = (
    (df["high"] < df[["open", "close"]].max(axis=1)) |
    (df["low"] > df[["open", "close"]].min(axis=1)) |
    (df["high"] < df["low"]) |
    (df["volume"] < 0)
)

print("\n===== DATA VALIDITY =====")
print("Invalid OHLC/volume rows:", invalid_ohlc.sum())

# 10. Display first and last records
print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== LAST 5 ROWS =====")
print(df.tail())
