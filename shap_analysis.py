
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

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

X_train = train_df[features]
y_train = train_df["Target"]

X_val = val_df[features]

# ==========================================
# 2. Train XGBoost
# ==========================================

model = XGBRegressor(
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

model.fit(X_train, y_train)

# ==========================================
# 3. Calculate SHAP values
# ==========================================

explainer = shap.TreeExplainer(model)

shap_values = explainer(X_val)

# ==========================================
# 4. Calculate mean absolute SHAP values
# ==========================================

mean_abs_shap = np.abs(
    shap_values.values
).mean(axis=0)

shap_results = pd.DataFrame({
    "Feature": features,
    "Mean_Absolute_SHAP": mean_abs_shap
})

shap_results = shap_results.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

print("\n===== SHAP FEATURE IMPORTANCE =====")
print(shap_results.round(6).to_string(index=False))

shap_results.to_csv(
    "outputs/model_results/shap_feature_importance.csv",
    index=False
)

# ==========================================
# 5. SHAP Summary Plot
# ==========================================

shap.summary_plot(
    shap_values,
    X_val,
    show=False
)

plt.tight_layout()

plt.savefig(
    "outputs/figures/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 6. SHAP Bar Plot
# ==========================================

shap.summary_plot(
    shap_values,
    X_val,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    "outputs/figures/shap_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSHAP analysis completed!")
print("SHAP results and plots saved successfully!")
