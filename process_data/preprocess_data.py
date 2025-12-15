import pandas as pd

# === 1️⃣ Đọc dữ liệu ===
bottle = pd.read_csv("/Users/buihung/NMKHDL/prj/data/bottle.csv", low_memory=False)
cast = pd.read_csv("/Users/buihung/NMKHDL/prj/data/cast.csv", low_memory=False)

# === 2️⃣ Chọn các cột phù hợp ===
bottle_cols = [
    'Cst_Cnt', 'Depthm', 'T_degC', 'Salnty',
    'O2ml_L', 'Oxy_µmol/Kg', 'O2Sat',
    'pH1', 'pH2', 'NH3uM', 'NO3uM', 'PO4uM', 'SiO3uM', 'ChlorA'
]
cast_cols = [
    'Cst_Cnt', 'Cruise_ID', 'Date', 'Year', 'Month',
    'Lat_Dec', 'Lon_Dec', 'Bottom_D'
]

# Chỉ giữ các cột có trong file thực tế
bottle = bottle[bottle.columns.intersection(bottle_cols)]
cast = cast[cast.columns.intersection(cast_cols)]

# === 3️⃣ Gộp hai bảng theo Cst_Cnt ===
merged = pd.merge(bottle, cast, on='Cst_Cnt', how='inner')

# === 4️⃣ Làm sạch dữ liệu ===
# Đổi tên cột để dễ hiểu
merged = merged.rename(columns={
    'Lat_Dec': 'Latitude',
    'Lon_Dec': 'Longitude',
    'Bottom_D': 'BottomDepth'
})

# Bỏ hàng thiếu tọa độ hoặc thời gian
merged = merged.dropna(subset=['Latitude', 'Longitude', 'Date'])

# Chuyển cột Date sang kiểu datetime
merged['Date'] = pd.to_datetime(merged['Date'], errors='coerce')

# Thay 'ND' hoặc giá trị lỗi bằng NaN
merged = merged.replace(['ND', 'NaN', -999], pd.NA)

# === 5️⃣ Xuất CSV đã làm sạch ===
merged.to_csv("calcofi_cleaned.csv", index=False)
print(f"✅ Xuất thành công {len(merged)} dòng dữ liệu CalCOFI đã làm sạch.")
