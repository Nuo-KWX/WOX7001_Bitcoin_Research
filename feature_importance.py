
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

# ==========================================
# 1. Load datasets
# ==========================================

train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/validation.csv")

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
# 2. Define models
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
# 3. Calculate permutation importance
# ==========================================

all_results = []

for name, model in models.items():

    print(f"\nAnalyzing {name}...")

    model.fit(X_train, y_train)

    importance = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="neg_mean_absolute_error",
        n_repeats=30,
        random_state=42,
        n_jobs=-1
    )

    model_results = pd.DataFrame({
        "Model": name,
        "Feature": features,
        "Importance_Mean": importance.importances_mean,
        "Importance_Std": importance.importances_std
    })

    model_results = model_results.sort_values(
        "Importance_Mean",
        ascending=False
    )

    print(model_results.round(6).to_string(index=False))

    all_results.append(model_results)

# ==========================================
# 4. Save combined results
# ==========================================

final_results = pd.concat(
    all_results,
    ignore_index=True
)

final_results.to_csv(
    "outputs/model_results/feature_importance.csv",
    index=False
)

print("\nFeature importance analysis completed!")
print("Results saved successfully!")
