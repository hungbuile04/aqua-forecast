import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib


def create_lag_features(df, target_cols, lags=(1, 4)):
    df = df.sort_values("Quarter").copy()

    for col in target_cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    return df

def train_model_with_station_history(csv_path, model_out_path):
    df = pd.read_csv(csv_path)

    target_cols = ["CN","As","Cd","Pb","Cu","Hg","Zn","Total_Cr"]

    for c in target_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- xử lý thời gian ----
    df["Quarter"] = pd.to_datetime(df["Quarter"])
    df["year"] = df["Quarter"].dt.year
    df["quarter"] = df["Quarter"].dt.quarter

    # ---- tạo lag theo từng trạm ----
    dfs = []
    for (x, y), g in df.groupby(["X", "Y"]):
        g_lag = create_lag_features(g, target_cols, lags=(1, 4))
        dfs.append(g_lag)

    df = pd.concat(dfs, ignore_index=True)

    # ---- feature & target ----
    feature_cols = (
        [f"{c}_lag1" for c in target_cols] +
        [f"{c}_lag4" for c in target_cols] +
        ["year", "quarter"]
    )

    df = df[feature_cols + target_cols].dropna()

    X = df[feature_cols]
    y = df[target_cols]

    # ---- model ----
    model = MultiOutputRegressor(
        xgb.XGBRegressor(
            n_estimators=800,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
    )

    model.fit(X, y)

    # ---- đánh giá train (tham khảo) ----
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred, multioutput="raw_values"))

    print("\n📊 RMSE (TRAIN):")
    for c, r in zip(target_cols, rmse):
        print(f"  {c:<10}: {r:.4f}")

    joblib.dump((model, feature_cols), model_out_path)
    print(f"\n✅ Saved model: {model_out_path}")

def predict_future_for_station(
    model_path,
    df_station,
    start_year,
    start_quarter,
    n_quarters
):
    target_cols = ["CN","As","Cd","Pb","Cu","Hg","Zn","Total_Cr"]

    model, feature_cols = joblib.load(model_path)

    df_station = df_station.copy()
    df_station["Quarter"] = pd.to_datetime(df_station["Quarter"])
    df_station = df_station.sort_values("Quarter")

    for c in target_cols:
        df_station[c] = pd.to_numeric(df_station[c], errors="coerce")

    # cần ít nhất 4 quý lịch sử
    history = df_station[target_cols].iloc[-4:].copy()

    results = []
    year, quarter = start_year, start_quarter

    for _ in range(n_quarters):
        row = {}

        for c in target_cols:
            row[f"{c}_lag1"] = float(history[c].iloc[-1])
            row[f"{c}_lag4"] = float(history[c].iloc[0])

        row["year"] = int(year)
        row["quarter"] = int(quarter)

        X_pred = pd.DataFrame([row])[feature_cols]

        # 🔒 ENSURE numeric 100%
        X_pred = X_pred.astype(float)

        y_pred = model.predict(X_pred)[0]

        result = {"year": year, "quarter": quarter}
        result.update(dict(zip(target_cols, y_pred)))
        results.append(result)

        # update history
        history = pd.concat(
            [history.iloc[1:], pd.DataFrame([y_pred], columns=target_cols)],
            ignore_index=True
        )

        quarter += 1
        if quarter > 4:
            quarter = 1
            year += 1
    
    df_future = pd.DataFrame(results)

    # Clip giá trị âm (ràng buộc vật lý)
    for c in target_cols:
        df_future[c] = df_future[c].clip(lower=0)

    return df_future

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = BASE_DIR.parent

    DATA_PATH = PROJECT_DIR / "data" / "data_quang_ninh" / "qn_env_clean_ready.csv"
    MODEL_PATH = PROJECT_DIR / "model" / "output" / "metal_ts_model.pkl"

    # ===== TRAIN =====
    train_model_with_station_history(DATA_PATH, MODEL_PATH)

    # ===== PREDICT cho 1 trạm =====
    df = pd.read_csv(DATA_PATH)
    df_station = df[(df["X"] == 2318587) & (df["Y"] == 428692)]

    df_future = predict_future_for_station(
        MODEL_PATH,
        df_station,
        start_year=2026,
        start_quarter=1,
        n_quarters=8   # 2 năm
    )

    print("\n🔮 Forecast:")
    print(df_future)

