import pandas as pd
import re
import os

# Tên file csv của bạn
current_dir = os.path.dirname(os.path.abspath(__file__))
# Ghép nối để tìm file csv nằm CÙNG THƯ MỤC với code
file_path = os.path.join(current_dir, 'qn_env_clean_ready.csv')

def fix_quarter_format():
    print(f"🔄 Đang đọc file: {file_path}")
    
    if not os.path.exists(file_path):
        print("❌ Không tìm thấy file csv.")
        return

    df = pd.read_csv(file_path)

    if 'Quarter' not in df.columns:
        print("❌ Không tìm thấy cột 'Quarter' trong file.")
        return

    print(f"   Dữ liệu mẫu trước khi sửa: {df['Quarter'].unique()[:3]}")

    # Hàm chuyển đổi: "Quý 1 2021" -> "2021-01-01"
    def convert_to_date(val):
        try:
            val_str = str(val).lower()
            # Regex bắt chữ "quý" + số + năm
            match = re.search(r'quý\s*(\d+).*?(\d{4})', val_str)
            if match:
                q = int(match.group(1))
                y = int(match.group(2))
                
                # Tính tháng đầu quý (Q1->1, Q2->4, Q3->7, Q4->10)
                m = (q - 1) * 3 + 1
                
                # Trả về format chuẩn YYYY-MM-DD
                return f"{y}-{m:02d}-01"
        except:
            pass
        return val # Nếu lỗi thì giữ nguyên để kiểm tra

    # Áp dụng chuyển đổi
    df['Quarter'] = df['Quarter'].apply(convert_to_date)

    # Lưu đè lại file CSV
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ Đã sửa xong! Dữ liệu được lưu đè vào: {file_path}")
    print(f"   Dữ liệu mẫu sau khi sửa:   {df['Quarter'].unique()[:3]}")
    print("👉 Bây giờ bạn có thể chạy Fine-tune bình thường mà không lo lỗi ngày tháng.")

if __name__ == "__main__":
    fix_quarter_format()