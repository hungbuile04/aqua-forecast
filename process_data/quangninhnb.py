# fine_tune_on_nb.py
# Requirements: pandas, numpy, scikit-learn, xgboost, joblib, pyproj (if need coord transform), geopandas (optional), sklearn

import os
import re
import glob
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ---------- CONFIG ----------
NB_CSV_PATH = "Tong hop NB.2021-2025.IN.csv"   # <-- đổi thành đường dẫn file NB của bạn
BASE_MODEL_PATH = "models/hk_quarterly_xgb_multi.joblib"  # model pre-trained trên HK
SAVE_DIR = "models_nb_finetune"
os.makedirs(SAVE_DIR, exist_ok=True)
# optional cluster config: set n_clusters=None to skip clustering
n_clusters = 4   # try 2-6; set None to skip
# target list to fine-tune (must be subset of base model targets)
TARGETS = ["DO","pH","NH4","PO4","TSS","Coliform","temp","salinity","BOD","COD","H2S"]
# ----------------------------

def parse_value_handle_lod(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        s = x.strip()
        if s == '':
            return np.nan
        if s.startswith('<'):
            try:
                lod = float(re.sub('[^0-9.eE+-]','', s))
                return lod/2.0
            except:
                return np.nan
        s2 = re.sub('[^0-9eE+-.]', '', s)
        try:
            return float(s2)
        except:
            return np.nan
    try:
        return float(x)
    except:
        return np.nan

def standardize_colname(c):
    s = c.strip().lower()
    s = re.sub(r'[^0-9a-z\+]+', '_', s)
    s = s.strip('_')
    return s

# 1) load base model
print("Loading base (HK) model from", BASE_MODEL_PATH)
saved = joblib.load(BASE_MODEL_PATH)
base_model = saved['model']   # MultiOutputRegressor wrapping XGB
base_scaler = saved['scaler']
base_features = saved['features']
base_targets = saved['targets']
print("Base model targets:", base_targets)

# 2) load NB data (assume CSV)
print("Loading NB data:", NB_CSV_PATH)
df = pd.read_csv(NB_CSV_PATH, low_memory=False)

# standardize columns
df.columns = [standardize_colname(c) for c in df.columns]
# try to find date column and station column
date_col = None
for c in df.columns:
    if 'date' in c:
        date_col = c
        break
if date_col is None:
    # maybe the file uses quarter string; try 'ky' or 'quarter' or 'quy'
    for c in df.columns:
        if 'quy' in c or 'quarter' in c or 'ky' in c:
            date_col = c
            break
if date_col is None:
    raise RuntimeError("Không tìm thấy cột ngày trong file NB. Cần cột date hoặc quarter.")
# parse to datetime; if data is quarter code like 'Q1 2024' you need to transform
try:
    df[date_col] = pd.to_datetime(df[date_col])
except:
    # attempt parse period-like 'Quý 1 2024' or 'T11/21' style -> user earlier had 'T11/21' maybe month/day
    # try to parse flexible
    def try_parse(s):
        import dateutil.parser as dp
        try:
            return dp.parse(str(s), dayfirst=False)
        except:
            return pd.NaT
    df[date_col] = df[date_col].apply(try_parse)

# find station column (NB)
station_col = None
for c in df.columns:
    if c.lower().startswith('nb') or 'station' in c or 'site' in c:
        station_col = c
        break
if station_col is None:
    # create a dummy station column if none
    df['station'] = 'NB'
    station_col = 'station'
else:
    df = df.rename(columns={station_col: 'station'})

df = df.rename(columns={date_col: 'date'})

# parse numeric columns with LOD handling
for c in df.columns:
    if c in ['station','date','lat','lon']:
        continue
    df[c] = df[c].apply(parse_value_handle_lod)

# 3) ensure we have the target columns (map possible alternative names)
col_map = {}
for c in df.columns:
    k = c.lower()
    if 'amoni' in k or 'nh4' in k or 'nh3' in k:
        col_map[c] = 'NH4'
    if 'asen' in k or 'as_' in k or k == 'as':
        col_map[c] = 'As'
    if 'colif' in k:
        col_map[c] = 'Coliform'
    if k == 'do' or 'oxy' in k:
        col_map[c] = 'DO'
    if 'dau' in k or 'oil' in k or 'mo' in k:
        col_map[c] = 'Oil'
    if 'fluor' in k or 'f-' in k or 'f' == k:
        col_map[c] = 'F'
    if 'mang' in k or 'mn' in k:
        col_map[c] = 'Mn'
    if 'phot' in k or 'po4' in k:
        col_map[c] = 'PO4'
    if 'sat' in k and 'sắt' not in k:  # language edge
        col_map[c] = 'salinity'
    if k in ['temp','temperature','nhiet_do','nhiệt']:
        col_map[c] = 'temp'
    if 'fe' == k or 'sắt' in k:
        col_map[c] = 'Fe'
    if 'tss' in k or 'tsl' in k:
        col_map[c] = 'TSS'
    if 'ph' == k or 'pH' in c:
        col_map[c] = 'pH'
    if 'bod' in k:
        col_map[c] = 'BOD'
    if 'cod' in k:
        col_map[c] = 'COD'
    if 'h2s' in k or 'h₂s' in k:
        col_map[c] = 'H2S'
# apply rename
df = df.rename(columns=col_map)

# 4) quarter aggregation like HK pipeline
df = df.dropna(subset=['date'])
df['quarter'] = df['date'].dt.to_period('Q').dt.to_timestamp()

# choose dynamic targets available in NB
available = [t for t in TARGETS if t in df.columns]
if len(available) == 0:
    raise RuntimeError("Không tìm thấy target nào hợp lệ trong file NB. Các cột có: " + ", ".join(df.columns))
print("Available NB targets:", available)

agg_funcs = {c: 'mean' for c in available}
if 'lat' in df.columns:
    agg_funcs['lat'] = 'first'
if 'lon' in df.columns:
    agg_funcs['lon'] = 'first'

quarterly_nb = df.groupby(['station','quarter']).agg(agg_funcs).reset_index()

# 5) optional clustering of stations to increase sample per group
if n_clusters is not None:
    # need lat/lon for clustering; if not available, cluster by station string (no-op)
    if 'lat' not in quarterly_nb.columns or 'lon' not in quarterly_nb.columns:
        print("No lat/lon -> skipping clustering.")
        do_cluster = False
    else:
        do_cluster = True
    if do_cluster:
        # compute station centroid lat/lon
        st_centroids = quarterly_nb.groupby('station')[['lat','lon']].first().reset_index()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        st_centroids['cluster'] = kmeans.fit_predict(st_centroids[['lat','lon']])
        # map back
        quarterly_nb = quarterly_nb.merge(st_centroids[['station','cluster']], on='station', how='left')
        # aggregate per cluster-quarter (mean)
        cluster_agg = quarterly_nb.groupby(['cluster','quarter']).mean().reset_index()
        # rename station->cluster label
        # for modeling we treat 'cluster' identifier as new 'station'
        cluster_agg['station'] = cluster_agg['cluster'].apply(lambda x: f"cluster_{x}")
        quarterly_nb = cluster_agg  # use aggregated
        print(f"Data aggregated to {n_clusters} clusters.")
else:
    print("Clustering skipped by config.")

# 6) create lag1 features for NB (same as HK pipeline)
numeric_cols = [c for c in available if c in quarterly_nb.columns]
if 'lat' in quarterly_nb.columns and 'lon' in quarterly_nb.columns:
    numeric_cols += ['lat','lon']
numeric_cols = list(dict.fromkeys(numeric_cols))
quarterly_nb = quarterly_nb.sort_values(['station','quarter'])
for col in numeric_cols:
    quarterly_nb[f"{col}_lag1"] = quarterly_nb.groupby('station')[col].shift(1)

quarterly_nb_model = quarterly_nb.dropna(subset=[f"{c}_lag1" for c in numeric_cols]).copy()
print("NB quarterly rows after lagging:", len(quarterly_nb_model))

# 7) Build features identical to base features where possible
# base_features from HK might include lagged set for variables HK had. We'll attempt to create same shaped X.
# create cyclical month features:
quarterly_nb_model['quarter_month'] = quarterly_nb_model['quarter'].dt.month
quarterly_nb_model['sin_month'] = np.sin(2*np.pi*quarterly_nb_model['quarter_month']/12)
quarterly_nb_model['cos_month'] = np.cos(2*np.pi*quarterly_nb_model['quarter_month']/12)

# build feature DataFrame with base_features intersection
X_nb = pd.DataFrame(index=quarterly_nb_model.index)
for f in base_features:
    if f in quarterly_nb_model.columns:
        X_nb[f] = quarterly_nb_model[f]
    else:
        # attempt to map by removing suffix "_lag1"
        if f.endswith('_lag1'):
            col0 = f[:-5]
            alt = f"{col0}_lag1"
            if alt in quarterly_nb_model.columns:
                X_nb[f] = quarterly_nb_model[alt]
            else:
                # fallback: set nan
                X_nb[f] = np.nan
        else:
            X_nb[f] = np.nan

# add sin/cos if in base features
if 'sin_month' in base_features and 'sin_month' in quarterly_nb_model.columns:
    X_nb['sin_month'] = quarterly_nb_model['sin_month']
if 'cos_month' in base_features and 'cos_month' in quarterly_nb_model.columns:
    X_nb['cos_month'] = quarterly_nb_model['cos_month']

# fill any remaining NaNs in X_nb with column median (safe)
X_nb = X_nb.fillna(X_nb.median())

# scale using base scaler (HK)
X_nb_scaled = base_scaler.transform(X_nb)

# 8) Get base predictions (HK model predictions) for NB
y_base_pred = base_model.predict(X_nb_scaled)  # shape (n_samples, n_targets)
# assemble Y_true for NB targets (only for targets present)
Y_true = quarterly_nb_model[[t for t in base_targets if t in quarterly_nb_model.columns]].values
# but y_base_pred columns align with base_targets order
# If NB is missing some base_targets, we will restrict residual training to intersection
common_targets = [t for t in base_targets if t in quarterly_nb_model.columns]
print("Common targets between base and NB:", common_targets)

# align arrays
# y_base_pred => columns order base_targets
# build y_base_common
idxs = [base_targets.index(t) for t in common_targets]
y_base_common = y_base_pred[:, idxs]
Y_true_common = quarterly_nb_model[common_targets].values

# compute residuals (true - base_pred)
residuals = Y_true_common - y_base_common

# 9) Train residual learner on NB (small model). We'll use MultiOutputRegressor(XGB) or RandomForest if few samples.
n_samples = X_nb_scaled.shape[0]
print("NB samples for residual training:", n_samples)
if n_samples < 30:
    # small dataset -> use simple model with low overfit risk
    base_resid_est = RandomForestRegressor(n_estimators=100, random_state=42)
else:
    base_resid_est = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)

resid_model = MultiOutputRegressor(base_resid_est)
resid_model.fit(X_nb_scaled, residuals)  # map features -> residuals

# 10) Build tuned predictions = base_pred + resid_pred
resid_pred = resid_model.predict(X_nb_scaled)
y_tuned = y_base_common + resid_pred

# 11) Evaluate tuned vs base on NB (if have true)
rmse_base = np.sqrt(np.mean((Y_true_common - y_base_common)**2, axis=0))
rmse_tuned = np.sqrt(np.mean((Y_true_common - y_tuned)**2, axis=0))
for i,t in enumerate(common_targets):
    print(f"{t}: RMSE_base={rmse_base[i]:.4f}, RMSE_tuned={rmse_tuned[i]:.4f}")

# overall
overall_base = np.sqrt(np.mean((Y_true_common - y_base_common)**2))
overall_tuned = np.sqrt(np.mean((Y_true_common - y_tuned)**2))
print(f"Overall RMSE base={overall_base:.4f}, tuned={overall_tuned:.4f}")

# 12) Save tuned ensemble
joblib.dump({
    "base_model": base_model,
    "resid_model": resid_model,
    "base_scaler": base_scaler,
    "common_targets": common_targets,
    "base_targets": base_targets,
    "base_features": base_features
}, os.path.join(SAVE_DIR, "hk_base_plus_resid_nb.joblib"))
print("Saved tuned ensemble to", os.path.join(SAVE_DIR, "hk_base_plus_resid_nb.joblib"))

# 13) Optional: export predictions with station/quarter
out = quarterly_nb_model[['station','quarter']].copy().reset_index(drop=True)
for i,t in enumerate(common_targets):
    out[f"{t}_base_pred"] = y_base_common[:, i]
    out[f"{t}_tuned_pred"] = y_tuned[:, i]
    out[f"{t}_true"] = Y_true_common[:, i]
out.to_csv(os.path.join(SAVE_DIR, "nb_tuned_predictions.csv"), index=False)
print("Exported predictions to", os.path.join(SAVE_DIR, "nb_tuned_predictions.csv"))
