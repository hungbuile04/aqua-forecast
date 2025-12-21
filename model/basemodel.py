import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import warnings
import os
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')


#  HÀU
OYSTER_FEATURES = [
    'DO', 'Temperature', 'pH', 'Salinity', 'NH3', 'H2S', 'BOD5', 
    'TSS', 'Coliform', 'Alkalinity', 'Transparency',
]

#  CÁ GIÒ
COBIA_FEATURES = [
    'DO', 'Temperature', 'pH', 'Salinity', 'NH3', 'PO4', 'BOD5', 
    'TSS', 'Coliform', 'Alkalinity', 'Transparency'
]


# Hàm đọc CSV, làm sạch, điền dữ liệu thiếu và tạo Lag Features.
def prepare_time_series_data(csv_path, features_list, lags=[1, 4]):
    csv_path = str(csv_path)
    df = pd.read_csv(csv_path)
    
    # Chỉ giữ lại các cột cần thiết
    keep_cols = ['Station', 'Quarter'] + [f for f in features_list if f in df.columns]
    df = df[keep_cols]

    # Xử lý thời gian (đổi thành thời gian theo quý)
    df['Date'] = pd.to_datetime(df['Quarter'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values(by=['Station', 'Date'])

    # Điền dữ liệu thiếu (Imputation) theo trạm (một số trạm bị thiếu quý, ví dụ thiếu 2020Q2 thì lấy trung bình của Q1 và Q3)
    def fill_missing(group):
        # Nội suy tuyến tính
        group[features_list] = group[features_list].interpolate(method='linear', limit_direction='both')
        # Fill median trạm
        group[features_list] = group[features_list].fillna(group[features_list].median())
        return group

    df = df.groupby('Station').apply(fill_missing).reset_index(drop=True)
    
    # Fill median cho các ô feature bị thiếu
    df[features_list] = df[features_list].fillna(df[features_list].median())

    # Tạo Lag Features
    lag_cols = []
    for col in features_list:
        for lag in lags:
            new_col_name = f"{col}_lag{lag}"
            lag_cols.append(new_col_name)
            df[new_col_name] = df.groupby('Station')[col].shift(lag)
    
    df['Quarter_Num'] = df['Date'].dt.quarter
    time_features = ['Quarter_Num']

    df_final = df.dropna().reset_index(drop=True)
    
    print(f"Kích thước dữ liệu train: {df_final.shape}")
    
    input_features = lag_cols + time_features
    return df_final, input_features


# Hàm huấn luyện
def train_forecast_model(csv_path, features, model_out_path, meta_out_path=None):
    model_out_path = str(model_out_path)
    
    df_train, input_cols = prepare_time_series_data(csv_path, features, lags=[1, 4])
    
    if df_train is None:
        return

    X = df_train[input_cols]      # Quá khứ
    y = df_train[features]        # Hiện tại (Mục tiêu)

    # Các tham số
    model = MultiOutputRegressor(xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,            # Độ sâu trung bình (tránh overfit)
        subsample=0.8,          # Mỗi cây học 80% số dòng
        colsample_bytree=0.8,   # Mỗi cây học 80% số cột, giống kiểu drop out trong NN
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42
    ))

    model.fit(X, y)
    
    # Tính RMSE sau khi train (dùng tập train để test nên là kết quả ko có ý nghĩa lắm)
    print("\n📊 KẾT QUẢ ĐÁNH GIÁ (TRAINING SCORE):")
    print("-" * 50)
    
    y_pred = model.predict(X)
    
    mse = mean_squared_error(y, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mse)
    
    for i, col_name in enumerate(features):
        print(f"   🔹 {col_name:<15} RMSE: {rmse[i]:.4f}")
        
    print("-" * 50)
    print(f"👉 RMSE trung bình toàn mô hình: {np.mean(rmse):.4f}")


    # Lưu model
    joblib.dump(model, model_out_path)
    print(f"\n🎉 Đã lưu model tại: {model_out_path}")

    # Lưu metadata
    if meta_out_path is None:
        meta_out_path = model_out_path.replace('.pkl', '_features.pkl')
    
    joblib.dump((input_cols, features), meta_out_path)
    print(f"ℹ️  Đã lưu danh sách features tại: {meta_out_path}")


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = BASE_DIR.parent 
    
    OUTPUT_DIR = PROJECT_DIR / "model" / "output"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    DATA_DIR = PROJECT_DIR / "data" / "hk_water_quality"


    train_forecast_model(
        csv_path = DATA_DIR / "hk_oyster_quarterly_21vars.csv",
        features = OYSTER_FEATURES,
        model_out_path = OUTPUT_DIR / "hk_oyster_forecast_model.pkl"
    )


    train_forecast_model(
        csv_path = DATA_DIR / "hk_cobia_quarterly_21vars.csv",
        features = COBIA_FEATURES,
        model_out_path = OUTPUT_DIR / "hk_cobia_forecast_model.pkl"
    )