import pandas as pd
import pickle as pkl
import numpy as np
from pathlib import Path

import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# ======================
# Global config
# ======================
SEED = 42
np.random.seed(SEED)

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ======================
# Load data
# ======================
with open(DATA_DIR / "CA_1_0.pkl", "rb") as f:
    df_train = pkl.load(f)

with open(DATA_DIR / "CA_1_1.pkl", "rb") as f:
    df_valid = pkl.load(f)

# ======================
# Split features / target
# ======================
TARGET = "sold"

y_train = df_train[TARGET]
y_valid = df_valid[TARGET]

X_train = df_train.drop(TARGET, axis=1)
X_valid = df_valid.drop(TARGET, axis=1)

# ======================
# Common params
# ======================
common_params = {
    "n_estimators": 1000,
    "learning_rate": 0.3,
    "max_depth": 8,
    "random_state": SEED
}

scores = {}
models = {}

# ======================
# 1️⃣ LightGBM
# ======================
lgbm_model = LGBMRegressor(
    **common_params,
    num_leaves=50,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=300,
    n_jobs=-1
)

lgbm_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    callbacks=[
        lgb.early_stopping(10),
        lgb.log_evaluation(1)
    ]
)

preds = lgbm_model.predict(X_valid)
mse = mean_squared_error(y_valid, preds)
rmse = np.sqrt(mse)

pkl.dump(lgbm_model, open(MODEL_DIR / "lgbm.pkl", "wb"))
scores["lgbm"] = rmse
models["lgbm"] = lgbm_model

print(f"LGBM RMSE: {rmse:.4f}")

# ======================
# 2️⃣ CatBoost
# ======================
cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.3,
    depth=8,
    loss_function="RMSE",
    early_stopping_rounds=10,
    verbose=1,
    random_state=SEED
)

cat_model.fit(X_train, y_train, eval_set=(X_valid, y_valid))

preds = cat_model.predict(X_valid)
mse = mean_squared_error(y_valid, preds)
rmse = np.sqrt(mse)

pkl.dump(cat_model, open(MODEL_DIR / "catboost.pkl", "wb"))
scores["catboost"] = rmse
models["catboost"] = cat_model

print(f"CatBoost RMSE: {rmse:.4f}")

# ======================
# 3️⃣ XGBoost
# ======================
xgb_model = XGBRegressor(
    **common_params,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    early_stopping_rounds=10
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=1
)

preds = xgb_model.predict(X_valid)
mse = mean_squared_error(y_valid, preds)
rmse = np.sqrt(mse)

pkl.dump(xgb_model, open(MODEL_DIR / "xgboost.pkl", "wb"))
scores["xgboost"] = rmse
models["xgboost"] = xgb_model

print(f"XGBoost RMSE: {rmse:.4f}")


# ======================
# ✅ Select & save best model
# ======================
best_model_name = min(scores, key=scores.get)
best_model = models[best_model_name]

pkl.dump(best_model, open(MODEL_DIR / "best_model.pkl", "wb"))

print("\n=====================")
print("Training completed")
print("Best model:", best_model_name)
print("Best RMSE:", scores[best_model_name])
print("=====================")