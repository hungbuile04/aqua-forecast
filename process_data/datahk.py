import os
import glob
import numpy as np
import pandas as pd

def parse_lod(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.strip()
        if x.startswith("<"):
            try:
                return float(x[1:]) / 2
            except:
                return np.nan
        try:
            return float(x)
        except:
            return np.nan
    return x

def normalize_colname(c):
    return (
        c.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )


COLUMN_MAP = {
    # bảng tiêu chí  → cột HK sau normalize
    "DO": "dissolved_oxygen_mgl",
    "Temperature": "temperature_°c",
    "pH": "ph",
    "Salinity": "salinity_psu",
    "NH3": "unionised_ammonia_mgl",
    "PO4": "orthophosphate_phosphorus_mgl",
    "BOD5": "5_day_biochemical_oxygen_demand_mgl",
    "TSS": "suspended_solids_mgl",
    "Coliform": "faecal_coliforms_cfu100ml"
}


def load_hk_water(data_dir):
    files = glob.glob(os.path.join(data_dir, "marine_water_quality_*.csv"))
    dfs = []

    for f in files:
        df = pd.read_csv(f)

        # normalize header
        df.columns = [normalize_colname(c) for c in df.columns]

        # rename bắt buộc
        df = df.rename(columns={
            "dates": "date",
            "station": "station",
            "depth": "depth"
        })

        # parse date
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # parse numeric
        for c in df.columns:
            if c not in ["date", "station", "depth"]:
                df[c] = df[c].apply(parse_lod)

        dfs.append(df)

    if not dfs:
        raise RuntimeError("❌ Không load được file HK nào")

    return pd.concat(dfs, ignore_index=True)

FINAL_COLUMNS = [
    "DO","Temperature","pH","Salinity","Alkalinity","Transparency",
    "NH3","H2S","PO4","BOD5","COD","Coliform","TSS",
    "CN","As","Cd","Pb","Cu","Hg","Zn","Total_Cr"
]

def aggregate_quarter(df, depth_value):
    depth_value = depth_value.lower()
    df = df[df["depth"].str.lower().str.contains(depth_value)].copy()

    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["quarter"] = df["date"].dt.to_period("Q").dt.to_timestamp()

    numeric_cols = [
        c for c in df.columns
        if c not in ["date", "station", "depth", "month", "quarter"]
    ]

    monthly = (
        df.groupby(["station", "month"])[numeric_cols]
        .mean()
        .reset_index()
    )

    monthly["quarter"] = monthly["month"].dt.to_period("Q").dt.to_timestamp()

    quarterly = (
        monthly.groupby(["station", "quarter"])[numeric_cols]
        .mean()
        .reset_index()
    )

    return quarterly


def standardize_schema(df):
    out = pd.DataFrame()
    out["Station"] = df["station"]
    out["Quarter"] = df["quarter"]

    for std_col, hk_col in COLUMN_MAP.items():
        if hk_col in df.columns:
            out[std_col] = df[hk_col]
        else:
            out[std_col] = np.nan

    # các trường không có trong HK
    for c in [
        "Alkalinity","Transparency","H2S","COD",
        "CN","As","Cd","Pb","Cu","Hg","Zn","Total_Cr"
    ]:
        out[c] = np.nan

    return out


def add_synthetic_h2s(df, mean=0.04, std=0.015):
    mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
    sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))

    h2s = np.random.lognormal(mu, sigma, size=len(df))
    h2s = np.clip(h2s, 0.0005, 0.06)

    df["H2S"] = h2s
    return df

def add_synthetic_alkalinity(
    df,
    mean=120,        # mg/L
    std=35,
    min_val=40,
    max_val=200
):
    """
    Sinh độ kiềm giả (mg/L) theo truncated normal
    """
    vals = np.random.normal(loc=mean, scale=std, size=len(df))
    vals = np.clip(vals, min_val, max_val)

    df["Alkalinity"] = vals
    df["Alkalinity_source"] = "synthetic_truncated_normal"
    return df

def add_synthetic_transparency(
    df,
    mean=35,        # cm
    std=12,
    min_val=20,
    max_val=80
):
    """
    Sinh độ trong giả (cm) – phù hợp bảng tiêu chí
    """
    mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
    sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))

    vals = np.random.lognormal(mean=mu, sigma=sigma, size=len(df))
    vals = np.clip(vals, min_val, max_val)

    df["Transparency"] = vals
    df["Transparency_source"] = "synthetic_lognormal"
    return df

if __name__ == "__main__":
    np.random.seed(42)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "water_data")
    )
    OUT_DIR = "prj/data/hk_water_quality"
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_hk_water(DATA_DIR)

    # ================= CÁ GIÒ (Middle) =================
    cobia_raw = aggregate_quarter(df, depth_value="middle")
    cobia_std = standardize_schema(cobia_raw)
    cobia_std = add_synthetic_h2s(cobia_std)
    cobia_std = add_synthetic_alkalinity(cobia_std)
    cobia_std = add_synthetic_transparency(cobia_std)

    cobia_std.to_csv(
        f"{OUT_DIR}/hk_cobia_quarterly_21vars.csv",
        index=False
    )

    # ================= HÀU (Surface) =================
    oyster_raw = aggregate_quarter(df, depth_value="surface")
    oyster_std = standardize_schema(oyster_raw)
    oyster_std = add_synthetic_h2s(oyster_std)
    oyster_std = add_synthetic_alkalinity(oyster_std)
    oyster_std = add_synthetic_transparency(oyster_std)

    oyster_std.to_csv(
        f"{OUT_DIR}/hk_oyster_quarterly_21vars.csv",
        index=False
    )

    print("✅ Generated files:")
    print(" - hk_cobia_quarterly_21vars.csv")
    print(" - hk_oyster_quarterly_21vars.csv")
