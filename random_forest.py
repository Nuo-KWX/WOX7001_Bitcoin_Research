import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
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

X_train = train_df[features]
y_train = train_df[target]

X_val = val_df[features]
y_val = val_df[target]


# ==========================================
# 3. Build Random Forest model
# ==========================================

model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 4. Train model
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 5. Make predictions
# ==========================================

y_pred = model.predict(X_val)


# ==========================================
# 6. Evaluate model
# ==========================================

mae = mean_absolute_error(y_val, y_pred)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))

r2 = r2_score(y_val, y_pred)

directional_accuracy = np.mean(
    np.sign(y_pred) == np.sign(y_val)
) * 100


# ==========================================
# 7. Display results
# ==========================================

print("\n===== RANDOM FOREST RESULTS =====")

print("MAE:", round(mae, 6))
print("RMSE:", round(rmse, 6))
print("R2 Score:", round(r2, 6))
print("Directional Accuracy:", round(directional_accuracy, 2), "%")


# ==========================================
# 8. Save predictions
# ==========================================

results = pd.DataFrame({
    "Date": val_df["date"],
    "Actual_Return": y_val,
    "Predicted_Return": y_pred
})

results.to_csv(
    "outputs/model_results/random_forest_predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")