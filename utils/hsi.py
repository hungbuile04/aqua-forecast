import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns

def compute_hsi(df_forecast, species):
    """
    Tính HSI cho forecast theo loài (oyster | cobia)

    Parameters
    ----------
    df_forecast : pd.DataFrame
        DataFrame dự báo (có các cột môi trường + kim loại)
    species : str
        "oyster" hoặc "cobia"

    Returns
    -------
    pd.DataFrame (thêm cột HSI, HSI_Level)
    """
    HSI_RULES = {
        "oyster": {
            "DO":          {"min_val": 5},
            "Temperature": {"low": 20, "high": 28},
            "pH":          {"low": 7.5, "high": 8.0},
            "Salinity":    {"low": 20, "high": 25},
            "Alkalinity":  {"low": 60, "high": 180},
            "Transparency":{"low": 20, "high": 50},
            "NH3":         {"max_val": 0.3},
            "H2S":         {"max_val": 0.05},
            "BOD5":        {"max_val": 50},
            "COD":         {"max_val": 150},
            "Coliform":    {"max_val": 5000},
            "TSS":         {"max_val": 50},
            "CN":          {"max_val": 0.1},
            "As":          {"max_val": 0.02},
            "Cd":          {"max_val": 0.005},
            "Pb":          {"max_val": 0.05},
            "Cu":          {"max_val": 0.2},
            "Hg":          {"max_val": 0.001},
            "Zn":          {"max_val": 0.5},
            "Total_Cr":    {"max_val": 0.1},
        },

        "cobia": {
            "DO":          {"min_val": 6},
            "Temperature": {"low": 24, "high": 28},
            "pH":          {"low": 8.0, "high": 8.5},
            "Salinity":    {"low": 27, "high": 33},
            "Alkalinity":  {"low": 60, "high": 180},
            "Transparency":{"low": 20, "high": 50},
            "NH3":         {"max_val": 0.1},
            "PO4":         {"max_val": 0.2},
            "BOD5":        {"max_val": 50},
            "COD":         {"max_val": 150},
            "Coliform":    {"max_val": 5000},
            "TSS":         {"max_val": 50},
            "CN":          {"max_val": 0.1},
            "As":          {"max_val": 0.02},
            "Cd":          {"max_val": 0.005},
            "Pb":          {"max_val": 0.05},
            "Cu":          {"max_val": 0.2},
            "Hg":          {"max_val": 0.001},
            "Zn":          {"max_val": 0.5},
            "Total_Cr":    {"max_val": 0.1},
        }
    }

    def _suitability_score(x, low=None, high=None, max_val=None, min_val=None):
        if pd.isna(x):
            return 0.0

        # Khoảng tối ưu
        if low is not None and high is not None:
            if x < low:
                return max(0.0, x / low)
            elif x > high:
                return max(0.0, (2 * high - x) / high)
            else:
                return 1.0

        # Càng nhỏ càng tốt
        if max_val is not None:
            return max(0.0, 1 - x / max_val)

        # Càng lớn càng tốt
        if min_val is not None:
            return min(1.0, x / min_val)

        return 0.0


    species = species.lower()
    if species not in HSI_RULES:
        raise ValueError("species phải là 'oyster' hoặc 'cobia'")

    rules = HSI_RULES[species]
    df = df_forecast.copy()

    for c in rules.keys():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    hsi_list = []

    for _, row in df.iterrows():
        scores = []

        for var, rule in rules.items():
            if var not in row:
                continue

            s = _suitability_score(
                row[var],
                low=rule.get("low"),
                high=rule.get("high"),
                max_val=rule.get("max_val"),
                min_val=rule.get("min_val"),
            )
            scores.append(s)

        hsi_list.append(np.mean(scores) if scores else 0.0)

    df["HSI"] = hsi_list

    # Gán nhãn mức độ phù hợp
    def _label(h):
        if h >= 0.85:
            return "Rất phù hợp"
        elif h >= 0.75:
            return "Phù hợp"
        elif h >= 0.5:
            return "Ít phù hợp"
        else:
            return "Không phù hợp"

    df["HSI_Level"] = df["HSI"].apply(_label)

    return df

#Test
BASE_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "data_quang_ninh" / "qn_env_clean_ready.csv"
model_path = PROJECT_DIR / "model" / "output" / "metal_ts_model.pkl"

# ===== COMPUTE HSI CHO TOÀN BỘ DỮ LIỆU VÀ TÍNH PHÂN PHỐI NHÃN HSI =====
df = pd.read_csv(DATA_PATH)

# Tính HSI (ví dụ cho 'oyster'); nếu muốn chuyên biệt cho 'cobia' đổi species
df_hsi = compute_hsi(df, species="cobia")

# Hiển thị vài hàng đầu để kiểm tra
print(df_hsi[["Station", "Quarter", "HSI", "HSI_Level"]].head())

# Tính phân phối nhãn HSI (counts + %)
counts = df_hsi["HSI_Level"].value_counts()
percent = df_hsi["HSI_Level"].value_counts(normalize=True) * 100
print("\nHSI Level counts:")
print(counts.to_string())
print("\nHSI Level percentages:")
for lvl, p in percent.items():
    print(f"  {lvl}: {p:.1f}%")

min_hsi = df_hsi["HSI"].min()
rows_min = df_hsi[df_hsi["HSI"] == min_hsi]
print(f"\nMin HSI = {min_hsi:.6f}")
print("Rows with min HSI:")
print(rows_min[["Station", "Quarter", "HSI", "HSI_Level"]].to_string(index=False))

max_hsi = df_hsi["HSI"].max()
rows_max = df_hsi[df_hsi["HSI"] == max_hsi]
print(f"\nMax HSI = {max_hsi:.6f}")
print("Rows with max HSI:")
print(rows_max[["Station", "Quarter", "HSI", "HSI_Level"]].to_string(index=False))


# ===== VẼ ĐỒ THỊ PHÂN PHỐI HSI =====

hsi_values = df_hsi["HSI"].dropna()
sigma_hsi = hsi_values.std()

plt.figure(figsize=(8, 5))

# Histogram + KDE
sns.histplot(
    hsi_values,
    bins=30,
    kde=True,
    stat="density",
    color="steelblue",
    edgecolor="black"
)

# Ngưỡng phân loại HSI
plt.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="HSI = 0.5")
plt.axvline(0.75, color="orange", linestyle="--", linewidth=1, label="HSI = 0.75")
plt.axvline(0.85, color="green", linestyle="--", linewidth=1, label="HSI = 0.85")

# Đánh dấu min / max
plt.axvline(min_hsi, color="red", linestyle=":", linewidth=1.5, label=f"Min HSI = {min_hsi:.3f}")
plt.axvline(max_hsi, color="purple", linestyle=":", linewidth=1.5, label=f"Max HSI = {max_hsi:.3f}")

plt.title("Distribution of HSI values (Oyster)", fontsize=13)
plt.xlabel("HSI")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

# Lưu ảnh
OUT_FIG = PROJECT_DIR / "figure"
OUT_FIG.mkdir(exist_ok=True)

fig_path = OUT_FIG / "hsi_distribution_cobia.png"
plt.savefig(fig_path, dpi=300)
plt.close()

print(f"\n📊 Saved HSI distribution plot to: {fig_path}")

# ===== ĐỒ THỊ SO SÁNH NGƯỠNG ΔHSI =====

plt.figure(figsize=(8, 4))

# Fake distance bins để minh hoạ (nếu chưa có summary thật)
# Nếu có summary thật (bin_center, mean_delta_hsi) thì thay vào
delta_hsi_example = np.sort(np.abs(hsi_values - hsi_values.mean()))

plt.plot(
    np.arange(len(delta_hsi_example)),
    delta_hsi_example,
    color="black",
    linewidth=1.5,
    label="|ΔHSI|"
)

plt.axhline(0.6 * sigma_hsi, color="green", linestyle="--", label="0.6σ threshold")

plt.xlabel("Sorted index (proxy for distance)")
plt.ylabel("|ΔHSI|")
plt.title("Comparison of ΔHSI thresholds")
plt.legend()
plt.tight_layout()

fig_path2 = OUT_FIG / "delta_hsi_threshold_comparison.png"
plt.savefig(fig_path2, dpi=300)
plt.close()

print(f"📊 Saved ΔHSI threshold comparison plot to: {fig_path2}")
