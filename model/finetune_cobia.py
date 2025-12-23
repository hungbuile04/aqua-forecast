import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import warnings
import os
from pathlib import Path
from sklearn.metrics import mean_squared_error

from basemodel import *

warnings.filterwarnings('ignore')

# Hàm fine-tune mô hình với dữ liệu mới
def finetune_model(base_model_path, new_data_path, output_path, features_list):
    base_model_path = str(base_model_path)
    output_path = str(output_path)
    
    if not os.path.exists(base_model_path):
        print(f"❌ Lỗi: Không tìm thấy file model gốc tại {base_model_path}")
        return

    model = joblib.load(base_model_path)
    print("✅ Đã load xong model gốc.")

    # LOAD METADATA (biết cấu trúc train dùng cột nào)
    meta_path = base_model_path.replace('.pkl', '_features.pkl')
    input_cols_old, features_old = joblib.load(meta_path)
    print("✅ Đã xác định được cấu trúc input/output cũ.")

    # Gọi hàm chuẩn bị dữ liệu từ basemodel
    print(f"🔄 Đang xử lý dữ liệu mới từ: {new_data_path}")
    df_ft, _ = prepare_time_series_data(new_data_path, features_list, lags=[1, 4])
    if df_ft is None or len(df_ft) == 0:
        print("⚠️ Dữ liệu fine-tune trống hoặc không đủ để tạo lag. Hủy bỏ.")
        return

    X_new = df_ft[input_cols_old]
    y_new = df_ft[features_list]

    print(f"📊 Kích thước dữ liệu Fine-tune: {len(X_new)} mẫu")

    # Fine-tune từng model con trong MultiOutputRegressor
    for i, estimator in enumerate(model.estimators_):
        target_name = features_list[i]
        old_booster = estimator.get_booster()
        
        estimator.set_params(learning_rate=0.005) 
        
        # gb_model=old_booster để tiếp tục từ model cũ
        estimator.fit(X_new, y_new.iloc[:, i], xgb_model=old_booster)
        
    # Đánh giá rmse
    print("\n📊 KẾT QUẢ SAU KHI FINE-TUNE (TRÊN TẬP DỮ LIỆU MỚI):")
    print("-" * 50)
    y_pred = model.predict(X_new)
    rmse = np.sqrt(mean_squared_error(y_new, y_pred, multioutput='raw_values'))
    
    for i, col_name in enumerate(features_list):
        print(f"   🔹 {col_name:<15} RMSE: {rmse[i]:.4f}")
    
    print("-" * 50)
    print(f"👉 RMSE trung bình: {np.mean(rmse):.4f}")

    # Lưu model fine-tune
    joblib.dump(model, output_path)
    joblib.dump((input_cols_old, features_list), output_path.replace('.pkl', '_features.pkl'))
    
    print(f"\n🎉 Đã lưu model Fine-tune tại: {output_path}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = BASE_DIR.parent
    
    MODEL_DIR = PROJECT_DIR / "model" / "output"
    BASE_COBIA_MODEL = MODEL_DIR / "hk_cobia_forecast_model.pkl"
    
    # Đường dẫn đến dữ liệu cần được fine-tune
    NEW_DATA_PATH = PROJECT_DIR / "data" / "data_quang_ninh" / "qn_env_clean_ready.csv"
    
    # Đường dẫn lưu model mới
    OUTPUT_FINETUNE = MODEL_DIR / "hk_cobia_finetuned.pkl"

    
    # Chạy Fine-tune cho CÁ GIÒ (sửa cái này để chạy lại cho HÀU)
    finetune_model(
        base_model_path = BASE_COBIA_MODEL,
        new_data_path = NEW_DATA_PATH,
        output_path = OUTPUT_FINETUNE,
        features_list = COBIA_FEATURES
    )