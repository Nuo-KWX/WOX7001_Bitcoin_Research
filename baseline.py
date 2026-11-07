import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Load datasets
train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/validation.csv")

y_train = train_df["Target"]
y_val = val_df["Target"]

# 2. Create baseline predictions

# Baseline 1: Predict zero return
zero_pred = np.zeros(len(y_val))

# Baseline 2: Predict historical mean return
mean_pred = np.full(len(y_val), y_train.mean())

# 3. Evaluation function
def evaluate_model(name, actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(mean_squared_error(actual, predicted))

    r2 = r2_score(actual, predicted)

    print(f"\n===== {name} =====")
    print("MAE:", round(mae, 6))
    print("RMSE:", round(rmse, 6))
    print("R2 Score:", round(r2, 6))


# 4. Evaluate both baselines
evaluate_model("ZERO RETURN BASELINE", y_val, zero_pred)

evaluate_model("HISTORICAL MEAN BASELINE", y_val, mean_pred)