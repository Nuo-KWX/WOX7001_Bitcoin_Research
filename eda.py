import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load processed dataset
df = pd.read_csv("data/processed/BTCUSDT_Model_Dataset.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"], utc=True)

print("Dataset shape:", df.shape)
print("\nDataset information:")
print(df.info())

print("\nDescriptive statistics:")
print(df[[
    "Return",
    "Volatility_7",
    "Volatility_30",
    "Volume_Change",
    "Price_Range",
    "MA_Gap",
    "Target"
]].describe())


# Bitcoin closing price trend
plt.figure(figsize=(12, 5))

plt.plot(df["date"], df["close"])

plt.title("Bitcoin Daily Closing Price (2021–2025)")
plt.xlabel("Date")
plt.ylabel("Closing Price (USDT)")
plt.grid(True)
plt.tight_layout()

plt.savefig("outputs/figures/bitcoin_price_trend.png")
plt.show()


# Daily return over time
plt.figure(figsize=(12, 5))

plt.plot(df["date"], df["Return"])

plt.title("Bitcoin Daily Log Returns (2021–2025)")
plt.xlabel("Date")
plt.ylabel("Daily Log Return")
plt.grid(True)
plt.tight_layout()

plt.savefig("outputs/figures/bitcoin_daily_returns.png")
plt.show()


# Bitcoin volatility
plt.figure(figsize=(12, 5))

plt.plot(df["date"], df["Volatility_7"], label="7-Day Volatility")
plt.plot(df["date"], df["Volatility_30"], label="30-Day Volatility")

plt.title("Bitcoin Market Volatility (2021–2025)")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("outputs/figures/bitcoin_volatility.png")
plt.show()


# Distribution of daily returns
plt.figure(figsize=(10, 5))

plt.hist(df["Return"], bins=50)

plt.title("Distribution of Bitcoin Daily Returns")
plt.xlabel("Daily Log Return")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()

plt.savefig("outputs/figures/return_distribution.png")
plt.show()



# ==========================================
# 8. Statistical Analysis
# ==========================================

from scipy.stats import skew, kurtosis, jarque_bera

returns = df["Return"]

# Skewness
return_skewness = skew(returns)

# Kurtosis
return_kurtosis = kurtosis(returns, fisher=False)

# Jarque-Bera normality test
jb_stat, jb_pvalue = jarque_bera(returns)

print("\n===== STATISTICAL TEST RESULTS =====")

print("Skewness:", return_skewness)
print("Kurtosis:", return_kurtosis)

print("\nJarque-Bera Test")
print("Test Statistic:", jb_stat)
print("P-value:", jb_pvalue)

if jb_pvalue < 0.05: # type: ignore
    print("Conclusion: Bitcoin returns are NOT normally distributed.")
else:
    print("Conclusion: Cannot reject the normal distribution assumption.")


# ==========================================
# 9. Volatility Clustering
# ==========================================

plt.figure(figsize=(12, 5))

plt.plot(df["date"], df["Return"].abs())

plt.title("Bitcoin Absolute Daily Returns (Volatility Clustering)")
plt.xlabel("Date")
plt.ylabel("Absolute Daily Return")
plt.grid(True)
plt.tight_layout()

plt.savefig("outputs/figures/bitcoin_volatility_clustering.png")
plt.show()
