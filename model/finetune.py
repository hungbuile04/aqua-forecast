import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# ======================================================
# CONFIG
# ======================================================

SPECIES = "oyster"   # "oyster" hoặc "cobia"

OYSTER_FEATURES = [
    'DO','Temperature','pH','Salinity','NH3','H2S',
    'BOD5','TSS','Coliform','Alkalinity','Transparency'
]

COBIA_FEATURES = [
    'DO','Temperature','pH','Salinity','NH3','PO4',
    'BOD5','TSS','Coliform','Alkalinity','Transparency'
]

FEATURES = OYSTER_FEATURES if SPECIES == "oyster" else COBIA_FEATURES

LAGS = [1, 4]        # t-1 và t-4 (1 quý, 1 năm)
SEED = 42

# ======================================================
# DATA PREP (GIỐNG HK)
# ======================================================

def prepare_time_series_data(csv_path, features, lags):
    df = pd.read_csv(csv_path)

    df['Date'] = pd.to_datetime(df['Quarter'])
    df = df.sort_values(['Station', 'Date'])

    # Nội suy theo trạm
    def fill_missing(g):
        g[features] = g[features].interpolate(
            method='linear', limit_direction='both'
        )
        g[features] = g[features].fillna(g[features].median())
        return g

    df = df.groupby('Station', group_keys=False).apply(fill_missing)

    # Lag features
    lag_cols = []
    for f in features:
        for lag in lags:
            col = f"{f}_lag{lag}"
            lag_cols.append(col)
            df[col] = df.groupby('Station')[f].shift(lag)

    # Feature thời gian
    df['Quarter_Num'] = df['Date'].dt.quarter

    df = df.dropna().reset_index(drop=True)

    X_cols = lag_cols + ['Quarter_Num']
    return df, X_cols


# ======================================================
# FINE-TUNE FUNCTION
# ======================================================

def finetune_from_hk(
    qn_csv,
    hk_model_path,
    out_model_path,
    features
):
    print("🔹 Loading HK base model...")
    base_model: MultiOutputRegressor = joblib.load(hk_model_path)

    df, X_cols = prepare_time_series_data(qn_csv, features, LAGS)

    X = df[X_cols]
    y = df[features]

    # =====================
    # Fine-tune từng output
    # =====================
    new_estimators = []

    for i, est in enumerate(base_model.estimators_):
        print(f"🔁 Fine-tuning target: {features[i]}")

        booster = est.get_booster()

        new_est = xgb.XGBRegressor(
            n_estimators=300,          # thêm cây mới
            learning_rate=0.03,        # nhỏ → fine-tune
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=SEED
        )

        new_est.fit(
            X, y.iloc[:, i],
            xgb_model=booster
        )

        new_estimators.append(new_est)

    finetuned_model = MultiOutputRegressor(
        estimator=None
    )
    finetuned_model.estimators_ = new_estimators

    # =====================
    # Đánh giá nhanh
    # =====================
    y_pred = finetuned_model.predict(X)
    rmse = np.sqrt(
        mean_squared_error(y, y_pred, multioutput='raw_values')
    )

    print("\n📊 RMSE sau fine-tune:")
    for f, r in zip(features, rmse):
        print(f"  {f:<15}: {r:.4f}")
    print(f"👉 RMSE trung bình: {rmse.mean():.4f}")

    joblib.dump(
        (finetuned_model, X_cols, features),
        out_model_path
    )
    print(f"\n✅ Saved fine-tuned model: {out_model_path}")


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    PROJECT_DIR = Path(__file__).resolve().parents[1]

    DATA_QN = PROJECT_DIR / "data" / "data_quang_ninh" / "qn_env_clean_ready.csv"
    MODEL_DIR = PROJECT_DIR / "model" / "output"
    MODEL_DIR.mkdir(exist_ok=True, parents=True)

    if SPECIES == "oyster":
        HK_MODEL = MODEL_DIR / "hk_oyster_forecast_model.pkl"
        OUT_MODEL = MODEL_DIR / "qn_oyster_finetuned.pkl"
    else:
        HK_MODEL = MODEL_DIR / "hk_cobia_forecast_model.pkl"
        OUT_MODEL = MODEL_DIR / "qn_cobia_finetuned.pkl"

    finetune_from_hk(
        qn_csv=DATA_QN,
        hk_model_path=HK_MODEL,
        out_model_path=OUT_MODEL,
        features=FEATURES
    )
