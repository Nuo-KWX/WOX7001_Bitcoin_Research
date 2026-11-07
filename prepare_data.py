import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. Load raw Bitcoin dataset
# ==========================================

input_path = Path("data/raw/binance/BTCUSDT_Daily_2021_2025.csv")

df = pd.read_csv(input_path)

print("Original dataset shape:", df.shape)


# ==========================================
# 2. Convert timestamps into dates
# ==========================================

timestamp = pd.to_numeric(df["open_time"], errors="coerce")

micro_mask = timestamp >= 1e14

dates = pd.Series(
    pd.to_datetime(timestamp, unit="ms", utc=True),
    index=df.index
)

dates.loc[micro_mask] = pd.to_datetime(
    timestamp.loc[micro_mask],
    unit="us",
    utc=True
)

df["date"] = dates


# ==========================================
# 3. Convert numerical columns
# ==========================================

numeric_columns = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_values("date").drop_duplicates(subset="date")


# ==========================================
# 4. Feature Engineering
# ==========================================

# Daily log return
df["Return"] = np.log(df["close"] / df["close"].shift(1))

# Historical returns
df["Lag_Return_1"] = df["Return"].shift(1)
df["Lag_Return_3"] = df["Return"].shift(3)
df["Lag_Return_7"] = df["Return"].shift(7)

# Market volatility
df["Volatility_7"] = df["Return"].rolling(7).std()
df["Volatility_30"] = df["Return"].rolling(30).std()

# Trading volume change
df["Volume_Change"] = df["volume"].pct_change()

# Daily price range
df["Price_Range"] = (df["high"] - df["low"]) / df["close"]

# Moving average gap
df["SMA_7"] = df["close"].rolling(7).mean()
df["MA_Gap"] = df["close"] / df["SMA_7"] - 1

# Next-day return (Target)
df["Target"] = df["Return"].shift(-1)

# Date of the target
df["Target_Date"] = df["date"].shift(-1)


# ==========================================
# 5. Clean missing and infinite values
# ==========================================

df = df.replace([np.inf, -np.inf], np.nan)

required_columns = [
    "Return",
    "Lag_Return_1",
    "Lag_Return_3",
    "Lag_Return_7",
    "Volatility_7",
    "Volatility_30",
    "Volume_Change",
    "Price_Range",
    "MA_Gap",
    "Target"
]

df = df.dropna(subset=required_columns)


# ==========================================
# 6. Save processed dataset
# ==========================================

output_path = Path("data/processed/BTCUSDT_Model_Dataset.csv")

df.to_csv(output_path, index=False)


# ==========================================
# 7. Display results
# ==========================================

print("\n===== FEATURE ENGINEERING COMPLETED =====")

print("Processed dataset shape:", df.shape)

print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())

print("\nFeatures:")
print(required_columns[:-1])

print("\nTarget:")
print("Target = Next-day Bitcoin daily return")

print("\nMissing values:")
print(df[required_columns].isnull().sum())

print("\nFirst 5 rows:")
print(df[["date", "Return", "Volatility_7", "Volatility_30", "Target"]].head())

print("\nSaved to:", output_path)