
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

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

# Combine training and validation data
train_val_df = pd.concat(
    [train_df, val_df],
    ignore_index=True
)

# ==========================================
# 2. Define features and target
# ==========================================

features = [
    "Return",
    "Lag_Return_1",
    "Lag_Return_3",
    "Lag_Return_7",
    "Volatility_7",
    "Volatility_30",
    "Volume_Change",
    "Price_Range",
    "MA_Gap"
]

target = "Target"

X_train = train_val_df[features]
y_train = train_val_df[target]

X_test = test_df[features]
y_test = test_df[target]

# ==========================================
# 3. Define models
# ==========================================

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror"
    )
}

# ==========================================
# 4. Define evaluation function
# ==========================================

results = []

def evaluate_model(name, actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    directional_accuracy = np.mean(
        np.sign(predicted) == np.sign(actual)
    ) * 100

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Directional Accuracy": directional_accuracy
    })

# ==========================================
# 5. Evaluate Baselines
# ==========================================

zero_pred = np.zeros(len(y_test))

mean_pred = np.full(
    len(y_test),
    y_train.mean()
)

evaluate_model(
    "Zero Return Baseline",
    y_test,
    zero_pred
)

evaluate_model(
    "Historical Mean Baseline",
    y_test,
    mean_pred
)

# ==========================================
# 6. Train and evaluate ML models
# ==========================================

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    evaluate_model(name, y_test, y_pred)

    # Save individual predictions
    prediction_df = pd.DataFrame({
        "Date": test_df["date"],
        "Actual_Return": y_test,
        "Predicted_Return": y_pred
    })

    filename = name.lower().replace(" ", "_")

    prediction_df.to_csv(
        f"outputs/model_results/final_{filename}_predictions.csv",
        index=False
    )

# ==========================================
# 7. Display final comparison
# ==========================================

results_df = pd.DataFrame(results)

print("\n===== FINAL TEST RESULTS =====")

print(
    results_df.round(6).to_string(index=False)
)

# ==========================================
# 8. Save comparison table
# ==========================================

results_df.to_csv(
    "outputs/model_results/final_model_comparison.csv",
    index=False
)

print("\nFinal results saved successfully!")
