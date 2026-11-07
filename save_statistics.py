import pandas as pd
from scipy.stats import skew, kurtosis, jarque_bera

# Load processed dataset
df = pd.read_csv("data/processed/BTCUSDT_Model_Dataset.csv")

# Select variables
variables = [
    "Return",
    "Volatility_7",
    "Volatility_30",
    "Volume_Change",
    "Price_Range",
    "MA_Gap"
]

# Calculate descriptive statistics
descriptive = df[variables].describe().T

descriptive = descriptive[
    ["count", "mean", "std", "min", "50%", "max"]
]

descriptive = descriptive.rename(columns={
    "count": "Count",
    "mean": "Mean",
    "std": "Std_Dev",
    "min": "Min",
    "50%": "Median",
    "max": "Max"
})

descriptive.to_csv(
    "outputs/model_results/descriptive_statistics.csv"
)

# Normality test
returns = df["Return"]

jb_stat, jb_pvalue = jarque_bera(returns)

normality = pd.DataFrame({
    "Metric": [
        "Skewness",
        "Kurtosis",
        "Jarque_Bera_Statistic",
        "P_Value"
    ],
    "Value": [
        skew(returns),
        kurtosis(returns, fisher=False),
        jb_stat,
        jb_pvalue
    ]
})

normality.to_csv(
    "outputs/model_results/normality_test.csv",
    index=False
)

print("===== DESCRIPTIVE STATISTICS =====")
print(descriptive.round(6))

print("\n===== NORMALITY TEST =====")
print(normality.to_string(index=False))

print("\nStatistics saved successfully!")