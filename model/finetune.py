import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ======================================================
# CONFIG
# ======================================================

SPECIES = "oyster"   # "oyster" hoặc "cobia"

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

FEATURES = OYSTER_FEATURES if SPECIES == "oyster" else COBIA_FEATURES

# ======================================================
# UTILS
# ======================================================

def impute_statistical(df, cols, method="median"):
    for c in cols:
        fill_value = df[c].median() if method == "median" else df[c].mean()
        df[c] = df[c].fillna(fill_value)
    return df


def clean_missing_pipeline(df, features):
    return impute_statistical(df, features, method="median")


def make_lag_features(df, features, lag=1):
    """
    Tạo X(t) từ X(t-1)
    """
    df = df.sort_values(["Station", "Quarter"])
    for f in features:
        df[f"{f}_lag{lag}"] = df.groupby("Station")[f].shift(lag)
    return df


def nearest_station(lat, lon, stations_df):
    """
    Tìm trạm gần nhất từ toạ độ click
    """
    d = (stations_df["lat"] - lat)**2 + (stations_df["lon"] - lon)**2
    return stations_df.loc[d.idxmin(), "Station"]

# ======================================================
# LOAD MODEL (HK → QN)
# ======================================================

if SPECIES == "oyster":
    MODEL_PATH = "model/output/hk_oyster_env_model.pkl"
    SCALER_PATH = "model/output/hk_oyster_scaler.pkl"
else:
    MODEL_PATH = "model/output/hk_cobia_env_model.pkl"
    SCALER_PATH = "model/output/hk_cobia_scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("✅ Loaded base HK model")

# ======================================================
# LOAD DATA
# ======================================================

# Dữ liệu QN theo quý
df_qn = pd.read_csv("data/qn_env_quarterly.csv")
df_qn["Quarter"] = pd.to_datetime(df_qn["Quarter"])

# Bảng toạ độ trạm
stations = pd.read_csv("data/stations.csv")

# ======================================================
# CHỌN TOẠ ĐỘ (GIẢ LẬP CLICK MAP)
# ======================================================

lat_click = 21.02
lon_click = 107.05

station = nearest_station(lat_click, lon_click, stations)
print(f"📍 Selected station: {station}")

df_station = df_qn[df_qn["Station"] == station].copy()

# ======================================================
# PREPROCESS
# ======================================================

df_station = clean_missing_pipeline(df_station, FEATURES)

# Tạo lag
df_lag = make_lag_features(df_station, FEATURES, lag=1)
df_lag = df_lag.dropna()

X = df_lag[[f"{f}_lag1" for f in FEATURES]]
y = df_lag[FEATURES]

# ======================================================
# FINE-TUNE MODEL ON QN
# ======================================================

Xs = scaler.transform(X)
model.fit(Xs, y)

print("✅ Fine-tuned model on Quảng Ninh data")

# ======================================================
# FORECAST 4 QUARTERS AHEAD
# ======================================================

last_row = df_station.sort_values("Quarter").iloc[-1]
x_t = last_row[FEATURES].values.reshape(1, -1)

future_preds = []
future_quarters = pd.date_range(
    start=last_row["Quarter"] + pd.offsets.QuarterBegin(),
    periods=4,
    freq="QS"
)

for q in future_quarters:
    x_scaled = scaler.transform(x_t)
    y_next = model.predict(x_scaled)

    future_preds.append(y_next.flatten())
    x_t = y_next   # recursive forecasting

# ======================================================
# OUTPUT
# ======================================================

df_forecast = pd.DataFrame(
    future_preds,
    columns=[f"{c}_pred" for c in FEATURES]
)
df_forecast["Quarter"] = future_quarters
df_forecast["Station"] = station

print("\n🔮 FORECAST RESULT (4 QUARTERS):")
print(df_forecast)

df_forecast.to_csv(
    f"output/forecast_{SPECIES}_{station}.csv",
    index=False
)

print("\n📁 Saved forecast file")
