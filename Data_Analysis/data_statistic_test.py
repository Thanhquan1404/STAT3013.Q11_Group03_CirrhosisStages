import pandas as pd
import numpy as np
import os
from scipy.stats import f_oneway, chi2_contingency
from typing import List

def run_statistical_tests(input_path: str,
                          output_path: str,
                          category_columns: List[str],
                          label_col: str):
    """
    Thực hiện kiểm định ANOVA (cho biến số) và Chi-square (cho biến phân loại)
    giữa các cột đặc trưng và biến mục tiêu (label column).

    Args:
        input_path (str): Đường dẫn đến file dữ liệu đã làm sạch.
        output_path (str): Đường dẫn để lưu kết quả kiểm định.
        category_columns (List[str]): Danh sách tên các cột phân loại (categorical).
        label_col (str): Tên cột biến mục tiêu (label).

    Returns:
        pd.DataFrame: DataFrame chứa kết quả của các kiểm định.
    """
    
    # 1. Step 1: Load data
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại đường dẫn {input_path}")
        return None
    
    # Tách cột số và cột phân loại
    all_features = df.drop(columns=[label_col]).columns
    
    # Xác định các cột số (numeric columns)
    # Loại bỏ các cột phân loại đã được chỉ định từ tất cả các cột đặc trưng
    numeric_cols = [col for col in all_features if col not in category_columns]
    
    # Danh sách để lưu trữ kết quả
    results = []
    
    # 2. ANOVA Test (Cho các cột số)
    print("--- Thực hiện ANOVA Test ---")
    
    for col in numeric_cols:
        # Lấy danh sách các giá trị của cột 'col' cho từng nhóm trong 'label_col'
        # GroupBy: Tách dữ liệu thành các nhóm dựa trên label_col
        # Get_group: Lấy mảng giá trị của cột 'col' cho từng nhóm
        groups = [df[df[label_col] == category][col].values for category in df[label_col].unique()]
        
        # Loại bỏ các nhóm rỗng hoặc có giá trị NaN (nếu có)
        groups = [g[~np.isnan(g)] for g in groups if len(g[~np.isnan(g)]) > 0]
        
        if len(groups) > 1: # Cần ít nhất 2 nhóm để thực hiện ANOVA
            f_statistic, p_value = f_oneway(*groups)
            results.append({
                'Feature': col,
                'Test': 'ANOVA',
                'Statistic': f_statistic,
                'P_Value': p_value,
                'Significance_Level': 'Significant' if p_value < 0.05 else 'Not Significant'
            })
            print(f"ANOVA | {col}: P-Value = {p_value:.4f}")
        else:
            print(f"Bỏ qua ANOVA cho {col}: Chỉ có 1 hoặc 0 nhóm dữ liệu hợp lệ.")

    # 3. Chi-square Test (Cho các cột phân loại)
    print("\n--- Thực hiện Chi-square Test ---")
    
    for col in category_columns:
        # Xây dựng bảng tần suất (Contingency Table)
        contingency_table = pd.crosstab(df[col], df[label_col])
        
        # Thực hiện Chi-square Test
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        
        results.append({
            'Feature': col,
            'Test': 'Chi-square',
            'Statistic': chi2,
            'P_Value': p_value,
            'Significance_Level': 'Significant' if p_value < 0.05 else 'Not Significant'
        })
        print(f"Chi-square | {col}: P-Value = {p_value:.4f}")

    # 4. Lưu kết quả
    results_df = pd.DataFrame(results)
    
    # Đảm bảo thư mục đầu ra tồn tại
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    results_df.to_csv(output_path, index=False)
    print(f"\nĐã lưu kết quả kiểm định thống kê thành công tại: {output_path}")
    
    return results_df

# --- Ví dụ cách sử dụng hàm (Dựa trên tên cột thường thấy trong dataset Indian Liver Patient) ---

statistical_results = run_statistical_tests(
    input_path='../Data/Preprocessed/indian_liver_patient_cleaned.csv',
    output_path='../calculation_result/indian_liver_patient_statistical_test_results.csv',
    category_columns=['Gender'],
    label_col='Result'
)

if statistical_results is not None:
    print("\n--- Kết quả kiểm định Thống kê ---")
    print(statistical_results)

statistical_results = run_statistical_tests(
    input_path='../Data/Preprocessed/liver_cirrhosis_cleaned.csv',
    output_path='../calculation_result/liver_cirrhosis_statistical_test_results.csv',
    category_columns=['Drug', 'Status', 'Ascites', 'Sex', 'Spiders', 'Edema'],
    label_col='Stage'
)

if statistical_results is not None:
    print("\n--- Kết quả kiểm định Thống kê ---")
    print(statistical_results)