from docx import Document
import pandas as pd
import re

# === Bước 1: Đọc bảng từ file DOCX ===
doc_path = "/Users/buihung/NMKHDL/prj/data/sea_2020.docx"
doc = Document(doc_path)

# Lấy bảng chính đầu tiên
table = doc.tables[0]

# Đọc toàn bộ dữ liệu
data = [[cell.text.strip() for cell in row.cells] for row in table.rows]

# Xác định header (3 dòng đầu)
header1, header2, header3 = data[:3]
rows = data[3:]

# Lấy danh sách Nhà bè (NB1–NB6) và Tide (CT/ĐT)
stations = header2[3:]
tides = header3[3:]

records = []

# === Bước 2: Làm phẳng dữ liệu theo dòng ===
current_metric = None
for row in rows:
    if len(row) < 4:
        continue
    # Nếu cột "Thông số" có giá trị mới → cập nhật thông số
    if row[1]:
        current_metric = row[1]

    thongso = current_metric
    kihieu = row[2]
    values = row[3:]

    for nb, tide, val in zip(stations, tides, values):
        val = val.replace(",", ".").strip() if val else None
        if not val or val == "ND":
            val = None
        records.append({
            "NB": nb,
            "Tide": tide,
            "Kí_hiệu": kihieu,
            "Thông_số": thongso,
            "value": val
        })

# Tạo DataFrame phẳng
df_long = pd.DataFrame(records)

# === Bước 3: Pivot để mỗi thông số thành 1 cột ===
df_wide = df_long.pivot_table(
    index=["NB", "Tide", "Kí_hiệu"],
    columns="Thông_số",
    values="value",
    aggfunc="first"
).reset_index()

# Làm phẳng tên cột (loại bỏ MultiIndex)
df_wide.columns.name = None

# === Bước 4: Xuất CSV ===
output_csv = "nuoc_bien_2020_clean.csv"
df_wide.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"✅ Đã lưu file CSV sạch: {output_csv}")
