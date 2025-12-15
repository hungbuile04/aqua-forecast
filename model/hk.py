# train_hk_water_quarterly.py
# Run in a notebook or script. Assumes a folder ./water_data with HK CSV files.

import os
import glob
import re
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# -------- CONFIG ----------
DATA_DIR = "data/water_data"   # path to folder with marine_water_quality_*.csv
DATETIME_COL_CANDIDATES = ["date", "Date", "sample_date", "sampledate"]
STATION_COL_CANDIDATES = ["station", "Station", "site", "Site", "station_id"]
TARGET_CANDIDATES = ["DO","pH","NH4","NH3","NH4+","PO4","PO4-3","TSS","Coliform",
                     "temp","temperature","salinity","BOD","COD","H2S"]
# final target name mapping we'll use
TARGET_MAP = {
    "do":"DO",
    "ph":"pH",
    "nh4":"NH4",
    "nh3":"NH4",        # map NH3 -> NH4 (we keep as ammonium)
    "nh4+":"NH4",
    "po4":"PO4",
    "po4_3":"PO4",
    "tss":"TSS",
    "coliform":"Coliform",
    "temperature":"temp",
    "temp":"temp",
    "salinity":"salinity",
    "bod":"BOD",
    "cod":"COD",
    "h2s":"H2S"
}
# --------------------------

def find_datetime_col(df):
    for c in DATETIME_COL_CANDIDATES:
        if c in df.columns:
            return c
    # otherwise try to find a datetime-like column
    for c in df.columns:
        if 'date' in c.lower():
            return c
    raise ValueError("No date column found")

def find_station_col(df):
    for c in STATION_COL_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        if 'stat' in c.lower() or 'site' in c.lower():
            return c
    return None

def standardize_colname(c):
    s = c.strip().lower()
    s = re.sub(r'[^0-9a-z\+]+', '_', s)
    s = s.strip('_')
    return s

def parse_value_handle_lod(x):
    # handle strings like "<0.05" -> 0.025 (LOD/2)
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.strip()
        if x == '':
            return np.nan
        if x.startswith('<'):
            try:
                lod = float(re.sub('[^0-9.eE+-]','', x))
                return lod / 2.0
            except:
                return np.nan
        # remove non-numeric text (e.g. "ND", "n/a")
        cleaned = re.sub('[^0-9eE+-.]', '', x)
        try:
            return float(cleaned)
        except:
            return np.nan
    try:
        return float(x)
    except:
        return np.nan

# --- 1. Read all files and concatenate ---
csv_files = glob.glob(os.path.join(DATA_DIR, "marine_water_quality_*.csv"))
if len(csv_files) == 0:
    raise FileNotFoundError(f"No files found in {DATA_DIR} matching marine_water_quality_*.csv")

frames = []
for fp in csv_files:
    df = pd.read_csv(fp, encoding='utf-8', low_memory=False)
    # standardize column names
    df.columns = [standardize_colname(c) for c in df.columns]
    # find date & station
    date_col = find_datetime_col(df)
    station_col = find_station_col(df)
    if station_col is None:
        # try adding filename as station prefix
        df['station'] = os.path.basename(fp).replace('.csv','')
        station_col = 'station'
    # parse date
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.rename(columns={date_col: 'date', station_col: 'station'})
    frames.append(df)

df = pd.concat(frames, ignore_index=True, sort=False)
print("Combined rows:", len(df), "columns:", df.shape[1])

# --- 2. Detect available target columns and standardize them ---
col_map = {}
for c in df.columns:
    key = c.lower().replace('-', '_').replace('+','')
    if key in TARGET_MAP:
        col_map[c] = TARGET_MAP[key]
    else:
        # try more liberal matching
        for k, v in TARGET_MAP.items():
            if k in key:
                col_map[c] = v
                break

# Only keep numeric-ish columns mapped to targets or common covariates
# parse values to numeric handling LOD
for c in list(df.columns):
    if c in ['date','station','lat','lon','geometry']:
        continue
    # check if column contains any '<' or non-numeric entries - then parse
    sample_vals = df[c].astype(str).sample(n=min(50, len(df)), random_state=1).tolist()
    need_parse = any(re.search(r'[<a-zA-Z]', str(x)) for x in sample_vals)
    if need_parse:
        df[c] = df[c].apply(parse_value_handle_lod)

# remap col names
df = df.rename(columns=col_map)
available_targets = [t for t in ["DO","pH","NH4","PO4","TSS","Coliform","temp","salinity","BOD","COD","H2S"] if t in df.columns]
print("Available dynamic targets/covariates detected:", available_targets)

# --- 3. Quarter aggregation per station ---
df = df.dropna(subset=['date'])
df['quarter'] = df['date'].dt.to_period('Q').dt.to_timestamp()  # quarter timestamp
agg_funcs = {c:'mean' for c in available_targets}
agg_funcs.update({'lat':'first','lon':'first'})  # if lat/lon exist
quarterly = df.groupby(['station','quarter']).agg(agg_funcs).reset_index()

# --- 4. Feature engineering: create lag-1 (previous quarter) features for each numeric col ---
numeric_cols = [c for c in available_targets if c in quarterly.columns]
numeric_cols += [c for c in ['lat','lon'] if c in quarterly.columns]
numeric_cols = list(dict.fromkeys(numeric_cols))  # unique keep order

quarterly = quarterly.sort_values(['station','quarter'])
for col in numeric_cols:
    quarterly[f"{col}_lag1"] = quarterly.groupby('station')[col].shift(1)

# Drop rows with NaNs in lag features (first quarter per station)
quarterly_model = quarterly.dropna(subset=[f"{c}_lag1" for c in numeric_cols]).copy()
print("Quarterly aggregated rows (after lagging):", len(quarterly_model))

# --- 5. Prepare X (features) and Y (targets) ---
# Select targets = dynamic environmental vars we want to predict next quarter.
# We'll predict the *current* quarter values using previous quarter features => to simulate t+1 prediction.
targets = [t for t in ["DO","pH","NH4","PO4","TSS","Coliform","temp","salinity","BOD","COD","H2S"] if t in quarterly_model.columns]
if len(targets) == 0:
    raise RuntimeError("No targets available. Check available columns in your data.")
print("Model targets:", targets)

# features: lagged numeric columns + quarter/month as cyclical + lat/lon
feature_cols = [f"{c}_lag1" for c in numeric_cols]
# add cyclical season features from quarter timestamp
quarterly_model['quarter_month'] = quarterly_model['quarter'].dt.month
quarterly_model['sin_month'] = np.sin(2*np.pi*quarterly_model['quarter_month']/12)
quarterly_model['cos_month'] = np.cos(2*np.pi*quarterly_model['quarter_month']/12)
feature_cols += ['sin_month','cos_month']
if 'lat' in quarterly_model.columns and 'lon' in quarterly_model.columns:
    feature_cols += ['lat','lon']

X = quarterly_model[feature_cols].astype(float)
Y = quarterly_model[targets].astype(float)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 6. Time-based train/test split: keep last full year (4 quarters) as test if possible ---
# find latest year in data
latest_quarter = quarterly_model['quarter'].max()
# test set = latest year (4 quarters)
test_start = (latest_quarter - pd.offsets.QuarterEnd(3)).to_timestamp()  # approx 4 quarters prior
mask_test = quarterly_model['quarter'] >= test_start
X_train, X_test = X_scaled[~mask_test.values], X_scaled[mask_test.values]
Y_train, Y_test = Y[~mask_test.values].values, Y[mask_test.values].values

print("Train rows:", X_train.shape[0], "Test rows:", X_test.shape[0])

# --- 7. Train MultiOutput XGBoost ---
xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, tree_method='auto')
model = MultiOutputRegressor(xgb)
model.fit(X_train, Y_train)

# --- 8. Evaluate ---
y_pred = model.predict(X_test)
results = {}
for i, t in enumerate(targets):
    rmse = mean_squared_error(Y_test[:,i], y_pred[:,i], squared=False)
    mae = mean_absolute_error(Y_test[:,i], y_pred[:,i])
    results[t] = {"rmse": float(rmse), "mae": float(mae)}
print("Evaluation on test set (per target):")
for t,v in results.items():
    print(f"{t}: RMSE={v['rmse']:.4f}, MAE={v['mae']:.4f}")

# overall aggregate
all_rmse = mean_squared_error(Y_test, y_pred, squared=False)
print(f"Overall multi-target RMSE: {all_rmse:.4f}")

# --- 9. Feature importances (aggregate by averaging the underlying estimators) ---
# XGBoost importances: average importance of each estimator in wrapper
importances = np.zeros(X_train.shape[1])
for est in model.estimators_:
    if hasattr(est, "feature_importances_"):
        importances += est.feature_importances_
importances /= len(model.estimators_)
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
print("Top feature importances:")
print(feat_imp.head(20))

# --- 10. Save model and scaler ---
os.makedirs("models", exist_ok=True)
joblib.dump({"model": model, "scaler": scaler, "features": feature_cols, "targets": targets}, "models/hk_quarterly_xgb_multi.joblib")
print("Model saved to models/hk_quarterly_xgb_multi.joblib")

# --- 11. Predict + attach back to quarterly_model for inspection ---
pred_df = quarterly_model[['station','quarter']].reset_index(drop=True).loc[mask_test.values].copy()
pred_df = pred_df.reset_index(drop=True)
pred_df[targets] = y_pred
pred_df['set'] = 'test_pred'
pred_df.to_csv("models/test_predictions_quarterly.csv", index=False)
print("Test predictions exported to models/test_predictions_quarterly.csv")
