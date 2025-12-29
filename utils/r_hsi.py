import numpy as np
import pandas as pd
import pathlib

def distance_vn2000_km(x1, y1, x2, y2):
    """
    Khoảng cách không gian cho hệ VN2000 (m → km)
    """
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2) / 1000.0

def compute_local_R_for_station_quarter(
    df_quarter,
    station_id,
    max_dist_km=20,
    bin_km=1.0
):
    """
    df_quarter: DataFrame của 1 (year, quarter)
                cột: station, x, y, hsi
    """
    center = df_quarter[df_quarter["station"] == station_id]
    if center.empty:
        return np.nan

    center = center.iloc[0]
    pairs = []

    for _, r in df_quarter.iterrows():
        if r["station"] == station_id:
            continue

        d = distance_vn2000_km(
            center.x, center.y,
            r.x, r.y
        )

        if d > max_dist_km:
            continue

        dhsi = abs(center.hsi - r.hsi)
        pairs.append((d, dhsi))

    if not pairs:
        return np.nan

    tmp = pd.DataFrame(pairs, columns=["dist_km", "delta_hsi"])

    # Gom theo khoảng cách
    bins = np.arange(0, max_dist_km + bin_km, bin_km)
    tmp["dist_bin"] = pd.cut(tmp["dist_km"], bins)

    summary = (
        tmp.groupby("dist_bin", observed=True)["delta_hsi"]
        .mean()
        .reset_index()
    )
    summary["bin_center"] = summary["dist_bin"].apply(lambda x: x.mid)

    delta_hsi_threshold = 0.2 * df_quarter["hsi"].std()

    exceed = summary[summary["delta_hsi"] >= delta_hsi_threshold]

    # Nếu chưa mất tương đồng trong phạm vi khảo sát
    if exceed.empty:
        return max_dist_km

    return exceed["bin_center"].min()

def compute_R_for_all_stations_all_quarters(
    hsi_csv_path,
    max_dist_km=50,
    bin_km=1.0
):
    """
    Input:
        hsi_csv_path: file hsi_oyster.csv hoặc hsi_cobia.csv

    Output:
        DataFrame: station, x, y, year, quarter, R_km
    """

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
                bin_km=bin_km
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
