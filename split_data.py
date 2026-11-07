import pandas as pd
from pathlib import Path

# 1. Load dataset
df = pd.read_csv("data/processed/BTCUSDT_Model_Dataset.csv")
df["date"] = pd.to_datetime(df["date"], utc=True)
df = df.sort_values("date").reset_index(drop=True)

# 2. Calculate split boundaries
n = len(df)
train_end = int(n * 0.70)
validation_end = int(n * 0.85)

# 3. Apply one-day purge at each boundary
train_df = df.iloc[:train_end - 1].copy()
validation_df = df.iloc[train_end:validation_end - 1].copy()
test_df = df.iloc[validation_end:].copy()

# 4. Display results
datasets = {
    "Training": train_df,
    "Validation": validation_df,
    "Testing": test_df
}

print("\n===== CHRONOLOGICAL SPLIT WITH PURGE =====")

for name, data in datasets.items():
    print(f"\n{name}:")
    print("Rows:", len(data))
    print("Date:", data["date"].min(), "to", data["date"].max())

# 5. Save datasets
output_dir = Path("data/processed")

train_df.to_csv(output_dir / "train.csv", index=False)
validation_df.to_csv(output_dir / "validation.csv", index=False)
test_df.to_csv(output_dir / "test.csv", index=False)

print("\nAll datasets saved successfully!")