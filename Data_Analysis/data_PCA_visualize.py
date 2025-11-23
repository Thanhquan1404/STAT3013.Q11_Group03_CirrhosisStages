import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def PCA_visualization_with_correlation(input_path="../Data/Preprocessed/indian_liver_patient_cleaned.csv",
                                       output_dir="../Figures/Data_PCA_Visualize/",
                                       target_column="Result",
                                       n_components=None):
    """
    PCA visualization with correlation matrix before and after PCA,
    including the analysis of PCA loadings (Ma trận Tải trọng).
    """
    # Load data
    df = pd.read_csv(input_path)
    
    # Drop Gender column if exists
    if "Gender" in df.columns:
        df = df.drop(columns=["Gender"])
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # --- Correlation matrix BEFORE PCA ---
    corr_before = df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_before, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix BEFORE PCA")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix_before_PCA.png"))
    plt.close()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    # Nếu n_components là None, PCA sẽ tính toán min(n_samples, n_features) thành phần
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Số lượng components thực tế được tạo
    n_actual_components = X_pca.shape[1]
    pc_cols = [f"PC{i+1}" for i in range(n_actual_components)]
    
    # --- Scree plot ---
    plt.figure(figsize=(8,5))
    explained_var_ratio = pca.explained_variance_ratio_
    plt.bar(range(1, len(explained_var_ratio)+1), explained_var_ratio*100, color='skyblue')
    plt.plot(range(1, len(explained_var_ratio)+1), np.cumsum(explained_var_ratio)*100, color='red', marker='o')
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained (%)")
    plt.title("Scree Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "PCA_scree_plot.png"))
    plt.close()
    
    # --- Scatter plot PC1 vs PC2 ---
    if n_actual_components >= 2:
        plt.figure(figsize=(8,6))
        scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='coolwarm', alpha=0.7)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA: PC1 vs PC2")
        plt.colorbar(scatter, label=target_column)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "PCA_PC1_PC2_scatter.png"))
        plt.close()
    
    # --- Correlation AFTER PCA ---
    # Convert X_pca to DataFrame
    df_pca = pd.DataFrame(X_pca, columns=pc_cols)
    df_pca[target_column] = y.values
    
    corr_after = df_pca.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_after, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix AFTER PCA")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix_after_PCA.png"))
    plt.close()

    # =========================================================================
    # --- PHÂN TÍCH MA TRẬN TẢI TRỌNG (LOADINGS MATRIX) ---
    # Các thành phần chính được lưu trong pca.components_
    # Đây là các vector riêng (eigenvectors)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Tạo DataFrame cho Ma trận Tải trọng
    # Hàng: Các biến gốc (Original Features)
    # Cột: Các Thành phần chính (Principal Components - PC)
    loadings_df = pd.DataFrame(loadings, 
                               columns=pc_cols, 
                               index=X.columns)
    
    print("\n--- MA TRẬN TẢI TRỌNG (LOADINGS MATRIX) ---")
    print("Giá trị thể hiện mức độ đóng góp/liên hệ của biến gốc với PC tương ứng.")
    print(loadings_df.head(10)) # In ra 10 hàng đầu tiên
    
    # Trực quan hóa Ma trận Tải trọng
    plt.figure(figsize=(n_actual_components*1.5, len(X.columns)/2))
    # Sử dụng cmap='RdBu_r' để thấy rõ tải trọng dương/âm (Red/Blue)
    sns.heatmap(loadings_df, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("PCA Loadings Matrix (Variables' contribution to PCs)")
    plt.ylabel("Original Feature")
    plt.xlabel("Principal Component")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_loadings_matrix.png"))
    plt.close()
    
    # --- Phân tích Tương quan của PC và các biến Gốc với Biến Mục tiêu ---
    # Lấy tương quan giữa PC và Result (cột cuối cùng của corr_after)
    pc_result_corr = corr_after.loc[pc_cols, target_column].sort_values(key=abs, ascending=False)
    
    print("\n--- TƯƠNG QUAN GIỮA PC VÀ BIẾN MỤC TIÊU (RESULT) ---")
    print(pc_result_corr)
    
    # Chọn PC có tương quan mạnh nhất (ví dụ: PC1)
    # Nếu PC1 là PC tương quan mạnh nhất với Result, ta xem xét tải trọng trên PC1.
    strongest_pc = pc_result_corr.index[0]
    
    print(f"\n--- PHÂN TÍCH TẢI TRỌNG CỦA PC MẠNH NHẤT: {strongest_pc} (Corr: {pc_result_corr.iloc[0]:.2f}) ---")
    # Xem xét các biến gốc đóng góp mạnh nhất vào PC tương quan mạnh nhất
    variable_contribution = loadings_df[strongest_pc].sort_values(key=abs, ascending=False)
    
    print("Các biến gốc đóng góp mạnh nhất vào PC này:")
    print(variable_contribution)
    
    # =========================================================================
    
    print(f"\nPCA visualization, correlation matrices, and loadings saved to {output_dir}")
    
    return X_pca, corr_before, corr_after, loadings_df

# Usage
# n_components=None để giữ tất cả thành phần và tính toán Loadings Matrix đầy đủ.
X_pca, corr_before, corr_after, loadings_df = PCA_visualization_with_correlation(
        input_path="../Data/Preprocessed/indian_liver_patient_cleaned.csv",
        output_dir="../Figures/Data_PCA_Visualize/indian_liver_patient",
        target_column="Result",
        n_components=None
)
X_pca, corr_before, corr_after, loadings_df = PCA_visualization_with_correlation(
        input_path="../Data/Preprocessed/liver_cirrhosis_cleaned.csv",
        output_dir="../Figures/Data_PCA_Visualize/liver_cirrhosis",
        target_column="Stage",
        n_components=None
)
