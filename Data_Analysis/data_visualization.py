import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import scipy.stats as stats

def data_visualization(input_path="../Data/Preprocessed/indian_liver_patient_cleaned.csv",
                       data_name: str="indian_liver_patient",
                       output_dir="../Figures/Data_Visualize/",
                       label_col="Result"):
    """
    Generate Boxplot+Violin, Distribution, Q-Q plots for each numeric feature
    on 2 classes (0=normal, 1=anomaly) in the same figure.

    Includes optional scaler (commented out for comparison).
    """

    # =====================
    # LOAD DATA
    # =====================
    df = pd.read_csv(input_path)

    # Remove Gender if exists
    if "Gender" in df.columns:
        df = df.drop(columns=["Gender"])

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found!")
    df = df.drop(columns=[label_col])


    os.makedirs(output_dir, exist_ok=True)

    # =====================
    # SCALING (OPTIONAL)
    # =====================
    # scaler = StandardScaler()
    # numeric_cols_to_scale = df.select_dtypes(include=['number']).columns.tolist()
    # numeric_cols_to_scale.remove(label_col)
    # df[numeric_cols_to_scale] = scaler.fit_transform(df[numeric_cols_to_scale])
    # print(">>> Scaling applied using StandardScaler")

    # NOTE:
    # Comment dòng scaler bên trên để so sánh dữ liệu trước/ sau khi scale

    # =====================
    # SELECT NUMERIC FEATURES
    # =====================

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if label_col in numeric_cols:
        numeric_cols.remove(label_col)

    print(f"Numeric features to visualize: {numeric_cols}")
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    for col in numeric_cols:
        data_dir = os.path.join(output_dir, data_name)
        feature_dir = os.path.join(data_dir, col)
        os.makedirs(feature_dir, exist_ok=True)

        # --- Boxplot + Violin ---
        plt.figure(figsize=(8,5))
        sns.boxplot(x=df[col], color='skyblue', width=0.4)
        sns.violinplot(x=df[col], color='lightgreen', alpha=0.5)
        plt.title(f"Boxplot & Violin Plot - {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(feature_dir, f"{col}_box_violin.png"))
        plt.close()
        
        # --- Distribution plot ---
        plt.figure(figsize=(8,5))
        sns.histplot(df[col], kde=True, bins=30, color='salmon')
        plt.title(f"Distribution Plot - {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(feature_dir, f"{col}_distribution.png"))
        plt.close()
        
        # --- Q-Q plot ---
        plt.figure(figsize=(6,6))
        stats.probplot(df[col], dist="norm", plot=plt)
        plt.title(f"Q-Q Plot - {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(feature_dir, f"{col}_qqplot.png"))
        plt.close()
        
        print(f"Plots generated for feature: {col}")

# Usage
# data_visualization(
#     input_path="../Data/Preprocessed/indian_liver_patient_cleaned.csv",
#     data_name="indian_liver_patient",
#     output_dir="../Figures/Data_Visualize",
#     label_col="Result",
# )
# data_visualization(
#     input_path="../Data/Preprocessed/liver_cirrhosis_cleaned.csv",
#     data_name="liver_cirrhosis",
#     output_dir="../Figures/Data_Visualize/",
#     label_col="Stage",
# )
