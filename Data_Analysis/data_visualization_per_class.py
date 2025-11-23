import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def data_visualization(input_path="../Data/Preprocessed/indian_liver_patient_cleaned.csv",
                       data_name: str="indian_liver_patient",
                       output_dir="../Figures/Data_Visualize_per_class/",
                       label_col="Result",
                       QQ_plot: bool=True):
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

    # =====================
    # VISUALIZATION
    # =====================
    for col in numeric_cols:
        data_dir = os.path.join(output_dir, data_name)
        feature_dir = os.path.join(data_dir, col)
        os.makedirs(feature_dir, exist_ok=True)

        # ============================
        # 1. BOX + VIOLIN (same figure)
        # ============================
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x=label_col, y=col, palette="Set2")
        sns.violinplot(data=df, x=label_col, y=col,
                       color="lightblue", alpha=0.4)
        plt.title(f"Boxplot + Violin - {col} (Class 0 vs 1)")
        plt.xlabel("Class (0=Normal, 1=Anomaly)")
        plt.tight_layout()
        plt.savefig(os.path.join(feature_dir, f"{col}_box_violin.png"))
        plt.close()

        # ============================
        # 2. DISTRIBUTION (same figure)
        # ============================
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x=col, hue=label_col,
                     kde=True, bins=30, palette="Set1",
                     alpha=0.35)
        plt.title(f"Distribution - {col} (0 vs 1)")
        plt.tight_layout()
        plt.savefig(os.path.join(feature_dir, f"{col}_distribution.png"))
        plt.close()

        if (QQ_plot):
            # ============================
            # 3. Q-Q PLOT (two subplots)
            # ============================
            class_values = df[label_col].unique()
            plt.figure(figsize=(10, 5))

            for i, cls in enumerate(class_values):
                plt.subplot(1, 2, i+1)
                sub_df = df[df[label_col] == cls][col].dropna()
                stats.probplot(sub_df, dist="norm", plot=plt)
                plt.title(f"Q-Q Plot - {col} (Class {cls})")

            plt.tight_layout()
            plt.savefig(os.path.join(feature_dir, f"{col}_qqplot.png"))
            plt.close()

        print(f"Generated: {col}")

    print("\n✨ DONE — All figures generated successfully!")

data_visualization(
    input_path="../Data/Preprocessed/indian_liver_patient_cleaned.csv",
    data_name="indian_liver_patient",
    output_dir="../Figures/Data_Visualize_per_class/",
    label_col="Result"
)
data_visualization(
    input_path="../Data/Preprocessed/liver_cirrhosis_cleaned.csv",
    data_name="liver_cirrhosis",
    output_dir="../Figures/Data_Visualize_per_class/",
    label_col="Stage",
    QQ_plot=False,
)