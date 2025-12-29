import pandas as pd
import pathlib
from forecast import (
    predict_future_metal_field_for_station,
    predict_future_non_metal_field_for_station,
    predict_for_station
)
from hsi import compute_hsi

def load_station_coordinates(csv_path):
    """
    Đọc file toạ độ trạm

    Expected columns
    ----------------
    maHieu, X, Y
    """
    df = pd.read_csv(csv_path)

    required = {"maHieu", "X", "Y"}
    if not required.issubset(df.columns):
        raise ValueError(f"File phải chứa các cột: {required}")

    df = df.rename(columns={
        "maHieu": "station",
        "X": "x",
        "Y": "y"
    })

    return df[["station", "x", "y"]]

def generate_hsi_for_species(
    coord_csv,
    species,
    start_year,
    start_quarter,
    n_quarters=4
):
    """
    Sinh bảng HSI cho tất cả trạm theo từng quý

    Returns
    -------
    DataFrame với schema:
    station, x, y, year, quarter, hsi
    """
    coords = load_station_coordinates(coord_csv)
    records = []

    for _, row in coords.iterrows():
        station = row["station"]
        x, y = row["x"], row["y"]

        # 1. Dự báo môi trường cho trạm
        df_forecast = predict_for_station(
            species=species,
            x=x,
            y=y,
            start_year=start_year,
            start_quarter=start_quarter,
            n_quarters=n_quarters
        )

        if df_forecast is None or df_forecast.empty:
            continue

        # 2. Tính HSI
        df_hsi = compute_hsi(df_forecast, species)

        # 3. Lưu kết quả
        for _, r in df_hsi.iterrows():
            records.append({
                "station": station,
                "x": x,
                "y": y,
                "year": int(r["year"]),
                "quarter": int(r["quarter"]),
                "hsi": float(r["HSI"])
            })

    return pd.DataFrame(records)

def generate_hsi_files(
    coord_csv,
    start_year,
    start_quarter,
    n_quarters=4,
    out_dir="."
):
    """
    Sinh 2 file:
    - hsi_oyster.csv
    - hsi_cobia.csv
    """

    # ===== HÀU =====
    df_oyster = generate_hsi_for_species(
        coord_csv=coord_csv,
        species="oyster",
        start_year=start_year,
        start_quarter=start_quarter,
        n_quarters=n_quarters
    )

    oyster_path = f"{out_dir}/hsi_oyster.csv"
    df_oyster.to_csv(oyster_path, index=False)

    # ===== CÁ GIÒ =====
    df_cobia = generate_hsi_for_species(
        coord_csv=coord_csv,
        species="cobia",
        start_year=start_year,
        start_quarter=start_quarter,
        n_quarters=n_quarters
    )

    cobia_path = f"{out_dir}/hsi_cobia.csv"
    df_cobia.to_csv(cobia_path, index=False)

    print("✅ Generated HSI files:")
    print(f" - {oyster_path}")
    print(f" - {cobia_path}")
    
BASE_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "data_quang_ninh" / "toa_do_qn.csv"
OUT_DIR = PROJECT_DIR / "data" / "data_quang_ninh"

generate_hsi_files(
    coord_csv=DATA_PATH,
    start_year=2026,
    start_quarter=1,
    n_quarters=4,
    out_dir=OUT_DIR
)
