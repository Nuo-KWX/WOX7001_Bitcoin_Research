
import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 1. Load datasets
# ==========================================

train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/validation.csv")
test_df = pd.read_csv("data/processed/test.csv")

train_val_df = pd.concat(
    [train_df, val_df],
    ignore_index=True
)

# ==========================================
# 2. Define volatility regimes
# ==========================================

threshold = train_val_df["Volatility_30"].median()

test_df["Regime"] = np.where(
    test_df["Volatility_30"] < threshold,
    "Low Volatility",
    "High Volatility"
)

print("Volatility Threshold:", round(threshold, 6))

# Check volatility regimes across all datasets

for name, data in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Testing", test_df)
]:

    high = (data["Volatility_30"] >= threshold).sum()
    low = (data["Volatility_30"] < threshold).sum()

    print(f"\n===== {name} REGIME DISTRIBUTION =====")
    print("Low Volatility:", low)
    print("High Volatility:", high)

# Display high-volatility test dates

print("\n===== HIGH VOLATILITY TEST DATES =====")

print(
    test_df.loc[
        test_df["Volatility_30"] >= threshold,
        ["date", "Volatility_30", "Target"]
    ].to_string(index=False)
)


print("\n===== TEST REGIME DISTRIBUTION =====")
print(test_df["Regime"].value_counts())

# ==========================================
# 3. Load model predictions
# ==========================================

prediction_files = {
    "Linear Regression":
        "outputs/model_results/final_linear_regression_predictions.csv",

    "Random Forest":
        "outputs/model_results/final_random_forest_predictions.csv",

    "XGBoost":
        "outputs/model_results/final_xgboost_predictions.csv"
}

predictions = {}

for name, file in prediction_files.items():
    predictions[name] = pd.read_csv(file)["Predicted_Return"].values

# Baseline predictions
predictions["Zero Return Baseline"] = np.zeros(len(test_df))

predictions["Historical Mean Baseline"] = np.full(
    len(test_df),
    train_val_df["Target"].mean()
)

# ==========================================
# 4. Evaluate each regime
# ==========================================

results = []

for regime in ["Low Volatility", "High Volatility"]:

    mask = test_df["Regime"] == regime

    actual = test_df.loc[mask, "Target"].values

    for name, all_predictions in predictions.items():

        predicted = all_predictions[mask.values]

        mae = mean_absolute_error(actual, predicted)

        rmse = np.sqrt(
            mean_squared_error(actual, predicted)
        )

        r2 = r2_score(actual, predicted)

        # Exclude zero predictions from directional accuracy
        nonzero = predicted != 0

        if nonzero.any():
            directional_accuracy = np.mean(
                np.sign(predicted[nonzero])
                == np.sign(actual[nonzero])
            ) * 100
        else:
            directional_accuracy = np.nan

        results.append({
            "Regime": regime,
            "Model": name,
            "Rows": len(actual),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Directional Accuracy": directional_accuracy
        })

# ==========================================
# 5. Display and save results
# ==========================================

results_df = pd.DataFrame(results)

print("\n===== VOLATILITY REGIME RESULTS =====")

print(results_df.round(6).to_string(index=False))

results_df.to_csv(
    "outputs/model_results/volatility_regime_results.csv",
    index=False
)

print("\nResults saved successfully!")
