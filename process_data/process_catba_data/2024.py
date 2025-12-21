from docx import Document
import pandas as pd

doc_path = "/Users/buihung/NMKHDL/prj/data/sea_2024.docx"  # thay đường dẫn thực tế
doc = Document(doc_path)
table = doc.tables[0]

# Đọc tất cả dòng trong bảng
data = [[cell.text.strip() for cell in row.cells] for row in table.rows]

# Xác định dòng chứa "pH" (hàng tiêu đề thực)
header_idx = next(i for i, row in enumerate(data) if any("pH" in c for c in row))
header = data[header_idx]  # đây là dòng thật chứa tên trường
rows = data[header_idx + 1:]

# Loại bỏ các dòng header lặp lại ("Vị trí", "KẾT QUẢ", v.v.)
clean_rows = [
    r for r in rows
    if not any(("Vị" in c or "vị" in c or "KẾT" in c or "kết" in c) for c in r)
]

# Tạo DataFrame
df = pd.DataFrame(clean_rows, columns=header)

# Chuẩn hóa tên cột
rename_map = {
    "Vị trí": "NB",
    "Đợt": "Tide",
    "pH": "pH",
    "Ôxy hòa tan (DO)\n(mg/l)": "DO",
    "Ôxy hòa tan (DO) (mg/l)": "DO",
    "Tổng chất rắn lơ lửng (TSS)\n(mg/l)": "TSS",
    "Tổng chất rắn lơ lửng (TSS) (mg/l)": "TSS",
    "Amôni \n(NH4+ tính theo N)\n(mg/l)": "NH4",
    "Phosphat \n(PO43- tính theo P)\n(mg/l)": "PO4",
    "Asen (As)\n(mg/l)": "As",
    "Chì \n(Pb)\n(mg/l)": "Pb",
    "Tổng dầu mỡ khoáng\n(mg/l)": "Dau_mo",
    "Coliform\n(MPN/\n100ml)": "Coliform",
}
df = df.rename(columns=lambda x: rename_map.get(x.strip(), x.strip()))

# Làm sạch dữ liệu
for col in df.columns:
    if col not in ["NB", "Tide"]:
        df[col] = (
            df[col]
            .astype(str)
            .replace(["ND", "nan", ""], None)
            .apply(lambda x: x.replace(",", ".") if isinstance(x, str) else x)
        )
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass

# Xuất CSV
output_csv = "nuoc_bien_2024_clean.csv"
df.to_csv(output_csv, index=False, encoding="utf-8-sig")
print("✅ Đã lưu file sạch:", output_csv)
