import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_indian_liver_data(
    input_path="../Raw/indian_liver_patient_preprocessed.csv",
    output_path="./indian_liver_patient_cleaned.csv"
):
    """
    Load, clean and save the Indian Liver Patient dataset.
    Steps:
    1. Load data from CSV
    2. Convert Result: merge class 2 into class 1 (1=has liver disease, 0=no disease)
    3. Fill missing values with column mean (only numeric columns)
    4. Remove duplicate rows
    5. Save cleaned data
    """
    # Step 1: Load data
    df = pd.read_csv(input_path)
    print(f"Original data shape: {df.shape}")
    print(f"Original class distribution:\n{df['Result'].value_counts(dropna=False)}")

    # Step 2: Fix the target variable - map both 1 and 2 → 1 (liver patient), others → 0
    # Common practice for this dataset: 1 = liver patient, 2 = non-liver patient → we merge 1 & 2 as positive? 
    # Actually: in the original dataset:
    #   1 = Patient has liver disease
    #   2 = No liver disease
    # So usually we map: 1→1, 2→0 (binary classification)

    # CORRECT MAPPING (standard for this dataset):
    df["Result"] = df["Result"].map({1: 1, 2: 0})

    # If any unexpected values (shouldn't be), turn them to NaN or handle
    # df["Result"] = df["Result"].map({1: 1, 2: 0}).fillna(0).astype(int)

    print(f"After mapping - class distribution:\n{df['Result'].value_counts()}")

    # Step 3: Fill missing values with mean (only numeric columns)
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # --- Fill missing categorical values with mode ---
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Step 4: Remove duplicate rows
    initial_shape = df.shape[0]
    df = df.drop_duplicates()
    print(f"Removed {initial_shape - df.shape[0]} duplicate rows")

    # Step 5: Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    print(f"Final cleaned data shape: {df.shape}")
    print(f"Final class distribution:\n{df['Result'].value_counts()}")

    return df


# Run the function
# df_cleaned = clean_indian_liver_data()

import pandas as pd
import numpy as np
import os

def clean_liver_cirrhosis_data(input_path="../Raw/liver_cirrhosis.csv",
                               output_path="./liver_cirrhosis_cleaned.csv"):
    """
    Thực hiện các bước làm sạch dữ liệu:
    1. Đọc dữ liệu.
    2. Điền giá trị thiếu (missing values) cho cột số bằng giá trị trung bình (mean).
    3. Điền giá trị thiếu cho cột phân loại (categorical) bằng giá trị mode.
    4. Xóa các hàng trùng lặp.
    5. Lưu dữ liệu đã làm sạch.
    
    Args:
        input_path (str): Đường dẫn đến file dữ liệu gốc.
        output_path (str): Đường dẫn để lưu file dữ liệu đã làm sạch.
        
    Returns:
        pd.DataFrame: DataFrame đã được làm sạch.
    """
    
    print(f"Bắt đầu làm sạch dữ liệu từ: {input_path}")
    
    # --- Step 1: Load data ---
    try:
        df = pd.read_csv(input_path)
        print(f"Đã tải {len(df)} hàng và {len(df.columns)} cột.")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại đường dẫn {input_path}")
        return None
    
    # Lưu lại kích thước ban đầu để so sánh
    initial_rows = len(df)
    
    # Xác định các cột số (numeric) và cột phân loại (categorical)
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # --- Step 2: Fill missing values with mean (only numeric columns) ---
    print("\n2. Xử lý giá trị thiếu (Missing Values)...")
    
    if len(numeric_cols) > 0:
        # Tính giá trị trung bình cho các cột số
        mean_values = df[numeric_cols].mean()
        # Điền giá trị thiếu bằng mean
        df[numeric_cols] = df[numeric_cols].fillna(mean_values)
        print(f"Đã điền giá trị thiếu cho {len(numeric_cols)} cột số bằng MEAN.")
    
    # --- Step 3: Fill missing categorical values with mode ---
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            # Tính giá trị mode (giá trị xuất hiện nhiều nhất)
            # [0] để lấy giá trị mode đầu tiên nếu có nhiều mode
            mode_value = df[col].mode()[0]
            # Điền giá trị thiếu bằng mode
            df[col] = df[col].fillna(mode_value)
        print(f"Đã điền giá trị thiếu cho {len(categorical_cols)} cột phân loại bằng MODE.")

    # --- Step 4: Remove duplicate rows ---
    print("\n4. Xóa các hàng trùng lặp...")
    
    # Tìm và xóa các hàng trùng lặp
    df_cleaned = df.drop_duplicates()
    rows_removed = initial_rows - len(df_cleaned)
    
    print(f"Đã xóa {rows_removed} hàng trùng lặp.")
    print(f"Kích thước dữ liệu cuối cùng: {len(df_cleaned)} hàng.")
    
    # --- Step 5: Save cleaned data ---
    # Đảm bảo thư mục đầu ra tồn tại
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    df_cleaned.to_csv(output_path, index=False)
    print(f"\n5. Đã lưu dữ liệu đã làm sạch thành công tại: {output_path}")
    
    return df_cleaned

# --- Thực thi hàm ---
df_cleaned = clean_liver_cirrhosis_data()
