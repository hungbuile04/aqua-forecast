import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns


def distance_vn2000_km(x1, y1, x2, y2):
    """
    Khoảng cách không gian cho hệ VN2000 (m → km)
    """
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2) / 1000.0

def hsi_label(h):
    if h >= 0.85:
        return "very_suitable"
    elif h >= 0.75:
        return "suitable"
    elif h >= 0.5:
        return "less_suitable"
    else:
        return "not_suitable"
    
HSI_LABEL_ORDER = {
    "not_suitable": 0,
    "less_suitable": 1,
    "suitable": 2,
    "very_suitable": 3
}


def compute_local_R_for_station_quarter(
    df_quarter,
    station_id,
    max_dist_km=30,
    bin_km=1.0,
    alpha=0.6
):
    # ---- 1. Lấy trạm trung tâm ----
    center = df_quarter[df_quarter["station"] == station_id]
    if center.empty:
        return np.nan
    center = center.iloc[0]

    center_hsi = center.hsi
    center_label = hsi_label(center_hsi)

    # ---- 2. Tính ngưỡng ΔHSI theo toàn quý ----
    sigma_hsi = df_quarter["hsi"].std()
    if pd.isna(sigma_hsi) or sigma_hsi == 0:
        return np.nan

    delta_hsi_threshold = alpha * sigma_hsi

    # ---- 3. Thu thập trạm lân cận ----
    records = []
    for _, r in df_quarter.iterrows():
        if r["station"] == station_id:
            continue

        d = distance_vn2000_km(center.x, center.y, r.x, r.y)
        if d > max_dist_km:
            continue

        records.append({
            "dist_km": d,
            "delta_hsi": abs(center_hsi - r.hsi),
            "label": hsi_label(r.hsi)
        })

    if not records:
        return np.nan

    tmp = pd.DataFrame(records)

    # ---- 4. Chia vòng đồng tâm theo khoảng cách ----
    bins = np.arange(0, max_dist_km + bin_km, bin_km)
    tmp["dist_bin"] = pd.cut(
        tmp["dist_km"],
        bins=bins,
        include_lowest=True
    )

    # ---- 5. Mở rộng R từng vòng, KHÔNG cho lẫn nhãn ----
    last_valid_R = 0.0

    for dist_bin, g in tmp.groupby("dist_bin", observed=True):
        if g.empty:
            continue

        # Điều kiện 1: ΔHSI trung bình vượt ngưỡng
        if g["delta_hsi"].mean() >= delta_hsi_threshold:
            break

        # Điều kiện 2 (QUAN TRỌNG): xuất hiện trạm khác nhãn trung tâm
        if (g["label"] != center_label).any():
            break

        last_valid_R = dist_bin.right

    # ---- 6. Ép R trong khoảng hợp lệ ----
    return min(max_dist_km, max(last_valid_R, 0.5))

def compute_R_for_all_stations_all_quarters(
    hsi_csv_path,
    max_dist_km=30,
    bin_km=1.0,
    alpha=0.6
):
    df = pd.read_csv(hsi_csv_path)

    required = {"station", "x", "y", "year", "quarter", "hsi"}
    if not required.issubset(df.columns):
        raise ValueError(f"File HSI phải có các cột: {required}")

    results = []

    for (year, quarter), g in df.groupby(["year", "quarter"]):
        g = g.reset_index(drop=True)

        for station in g["station"].unique():
            R = compute_local_R_for_station_quarter(
                df_quarter=g,
                station_id=station,
                max_dist_km=max_dist_km,
                bin_km=bin_km,
                alpha=alpha
            )

            row = g[g["station"] == station].iloc[0]

            results.append({
                "station": station,
                "x": row.x,
                "y": row.y,
                "year": int(year),
                "quarter": int(quarter),
                "R_km": R
            })

    return pd.DataFrame(results)



BASE_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "data_quang_ninh" / "toa_do_qn.csv"
OUT_DIR = PROJECT_DIR / "data" / "data_quang_ninh"

# Cho hàu
df_R_oyster = compute_R_for_all_stations_all_quarters(
    hsi_csv_path=OUT_DIR / "hsi_oyster.csv",
)
df_R_oyster.to_csv(OUT_DIR / "R_oyster.csv", index=False)

# Cho cá giò
df_R_cobia = compute_R_for_all_stations_all_quarters(
    hsi_csv_path=OUT_DIR / "hsi_cobia.csv",
)
df_R_cobia.to_csv(OUT_DIR / "R_cobia.csv", index=False)