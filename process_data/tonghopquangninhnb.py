import pandas as pd
import numpy as np

# =========================
# SYNTHETIC FUNCTIONS
# =========================
def add_synthetic_h2s(df, mean=0.04, std=0.015, seed=42):
    rng = np.random.default_rng(seed)
    mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
    sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))

    h2s = rng.lognormal(mu, sigma, size=len(df))
    h2s = np.clip(h2s, 0.0005, 0.06)

    df["H2S"] = h2s
    return df


def add_synthetic_cod(
    df,
    mean=90,
    std=60,
    min_val=0.5,
    max_val=220,
    seed=42
):
    rng = np.random.default_rng(seed)

    mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
    sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))

    cod = rng.lognormal(mean=mu, sigma=sigma, size=len(df))
    cod = np.clip(cod, min_val, max_val)

    df["COD"] = cod
    return df

def add_synthetic_bod5(
    df,
    mean=35,        # trung bình quan sát
    std=18,         # độ phân tán
    min_val=0.3,    # có thể rất nhỏ
    max_val=60,     # cho phép vượt nhẹ QCVN
    seed=42
):
    """
    Sinh dữ liệu BOD5 giả (mg/L) theo phân phối lognormal
    """
    rng = np.random.default_rng(seed)

    # chuyển mean/std sang tham số lognormal
    mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
    sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))

    bod5 = rng.lognormal(mean=mu, sigma=sigma, size=len(df))
    bod5 = np.clip(bod5, min_val, max_val)

    df["BOD5"] = bod5
    return df


def add_synthetic_alkalinity(
    df,
    mean=120,       # mg/L
    std=40,
    min_val=40,
    max_val=200,
    seed=42
):
    """
    Sinh dữ liệu độ kiềm giả (mg/L) theo truncated normal
    """
    rng = np.random.default_rng(seed)

    alk = rng.normal(loc=mean, scale=std, size=len(df))
    alk = np.clip(alk, min_val, max_val)

    df["Alkalinity"] = alk
    return df



# =========================
# LOAD DATA
# =========================
df = pd.read_excel("data/data_quang_ninh/Tong-hop-21-24-gen.xlsx")
coords = pd.read_csv("data/data_quang_ninh/toa_do_qn.csv")  # cột: mahieu, lat, lon

# =========================
# SELECT + RENAME COLUMNS
# =========================
df = df.rename(columns={
    "KHM": "Station",
    "Quý": "Quarter",
    "Nhiệt độ": "Temperature",
    "pH": "pH",
    "DO": "DO",
    "Độ muối": "Salinity",
    "Amoni": "NH3",
    "Phosphat": "PO4",
    "TSS": "TSS",
    "Coliform": "Coliform",
    "Độ trong": "Transparency",
    "Tổng dầu, mỡ khoáng": "COD",
    "Tổng xianua": "CN",
    "As": "As",
    "Cd": "Cd",
    "Pb": "Pb",
    "Cu": "Cu",
    "Hg": "Hg",
    "Zn": "Zn",
    "Cr": "Total_Cr"
})

# Chỉ giữ cột cần thiết
KEEP_COLS = [
    "Station","Quarter","DO","Temperature","pH","Salinity",
    "NH3","PO4","TSS","Coliform","Transparency",
    "COD","CN","As","Cd","Pb","Cu","Hg","Zn","Total_Cr"
]
df = df[KEEP_COLS]

# =========================
# JOIN TOẠ ĐỘ
# =========================
coords = coords.rename(columns={"maHieu": "Station"})
df = df.merge(coords, on="Station", how="left")

# =========================
# ADD SYNTHETIC FEATURES
# =========================
df = add_synthetic_h2s(df)
df = add_synthetic_cod(df)
df = add_synthetic_bod5(df)
df = add_synthetic_alkalinity(df)

# =========================
# FINAL OUTPUT
# =========================
df.to_csv("data/data_quang_ninh/qn_env_clean_ready.csv", index=False)

print("✅ Done: qn_env_clean_ready.csv")
