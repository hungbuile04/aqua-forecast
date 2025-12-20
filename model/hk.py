from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
import os
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

# ================= CONFIG =================
OYSTER_FEATURES = [
    "DO","Temperature","pH","Salinity",
    "Alkalinity","Transparency",
    "NH3","H2S","BOD5","Coliform","TSS"
]

COBIA_FEATURES = [
    "DO","Temperature","pH","Salinity",
    "Alkalinity","Transparency",
    "NH3","PO4","BOD5","Coliform","TSS"
]

def impute_statistical(df, cols, method="median"):
    for c in cols:
        if method == "median":
            fill_value = df[c].median()
        elif method == "mean":
            fill_value = df[c].mean()
        else:
            raise ValueError("method must be mean or median")

        df[c] = df[c].fillna(fill_value)

    return df


def clean_missing_pipeline(df, ENV_COLS):
    df = impute_statistical(
        df,
        cols=[c for c in ENV_COLS if c in df.columns],
        method="median"
    )

    return df

def clip_percentile(series, lower=0.01, upper=0.99):
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)

def handle_outliers(df, features):
    df = df.copy()

    # ===== log-scale variables =====
    log_cols = ["Coliform", "TSS", "BOD5", "NH3"]
    for c in log_cols:
        if c in features and c in df.columns:
            df[c] = np.log10(df[c] + 1)
            df[c] = clip_percentile(df[c], 0.01, 0.99)

    return df

# ================= TRAIN FUNCTION =================
def train_model(
    csv_path,
    features,
    model_out,
    scaler_out
):
    df = pd.read_csv(csv_path)

    # ⚠️ KHÔNG dropna cứng nếu đã impute
    df = clean_missing_pipeline(df, features)

    X = df[features]
    y = df[features]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    )
    # model = MultiOutputRegressor(
    #             RandomForestRegressor(
    #                 n_estimators=300,
    #                 max_depth=12,
    #                 random_state=42,
    #                 n_jobs=-1
    #             )
    #         )

    model.fit(Xs, y)

    # ===== TÍNH LOSS =====
    y_pred = model.predict(Xs)

    rmse_per_feature = np.sqrt(
        mean_squared_error(y, y_pred, multioutput="raw_values")
    )
    rmse_mean = rmse_per_feature.mean()

    print("📉 Training RMSE per feature:")
    for f, r in zip(features, rmse_per_feature):
        print(f"  {f:15s}: {r:.4f}")

    print(f"📉 Mean Training RMSE: {rmse_mean:.4f}")

    # ===== SAVE =====
    joblib.dump(model, model_out)
    joblib.dump(scaler, scaler_out)

    print(f"✅ Saved model: {model_out}")

# ================= MAIN =================
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = BASE_DIR.parent
    OUTPUT_DIR = PROJECT_DIR / "model" / "output"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ===== PATHS =====
    DATA_DIR = PROJECT_DIR / "data" / "hk_water_quality"

    # ===== HÀU =====
    train_model(
        csv_path=DATA_DIR / "hk_oyster_quarterly_21vars.csv",
        features=OYSTER_FEATURES,
        model_out=OUTPUT_DIR / "hk_oyster_env_model.pkl",
        scaler_out=OUTPUT_DIR / "hk_oyster_scaler.pkl"
    )

    # ===== CÁ GIÒ =====
    train_model(
        csv_path=DATA_DIR / "hk_cobia_quarterly_21vars.csv",
        features=COBIA_FEATURES,
        model_out=OUTPUT_DIR / "hk_cobia_env_model.pkl",
        scaler_out=OUTPUT_DIR / "hk_cobia_scaler.pkl"
    )
