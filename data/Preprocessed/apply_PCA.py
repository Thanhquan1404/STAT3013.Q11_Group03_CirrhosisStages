import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

def apply_PCA_1D(input_path: str, 
                 output_path: str, 
                 target_column: str = "Result"):
    """
    Apply PCA (1 component) to a dataset and save output CSV with columns: Data, Result
    """
    # Load data
    df = pd.read_csv(input_path)
    
    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA with 1 component
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create output DataFrame
    df_pca = pd.DataFrame({
        "Data": X_pca.flatten(),  # flatten to 1D
        "Result": y
    })
    
    # Create output directory if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df_pca.to_csv(output_path, index=False)
    print(f"PCA applied. Output saved to {output_path} with shape {df_pca.shape}")
    
    return df_pca

# Apply PCA for raw data for visualize
# apply_PCA_1D(
#     input_path="indian_liver_patient_cleaned.csv",
#     output_path="/indian_liver_patient_cleaned_after_PCA.csv"
# )
# Apply PCA for KFold data after SMOTE
# KFold = 1
# apply_PCA_1D(
#     input_path="../data_apply_SMOTE/KFold_data/indian_liver_patient_train_k_fold_01_SMOTE.csv",
#     output_path="../data_apply_SMOTE/PCA/indian_liver_patient_train_k_fold_01_PCA.csv"
# )
# KFold = 2
# apply_PCA_1D(
#     input_path="../data_apply_SMOTE/KFold_data/indian_liver_patient_train_k_fold_02_SMOTE.csv",
#     output_path="../data_apply_SMOTE/PCA/indian_liver_patient_train_k_fold_02_PCA.csv"
# )
# KFold = 3
# apply_PCA_1D(
#     input_path="../data_apply_SMOTE/KFold_data/indian_liver_patient_train_k_fold_03_SMOTE.csv",
#     output_path="../data_apply_SMOTE/PCA/indian_liver_patient_train_k_fold_03_PCA.csv"
# )
# KFold = 4
# apply_PCA_1D(
#     input_path="../data_apply_SMOTE/KFold_data/indian_liver_patient_train_k_fold_04_SMOTE.csv",
#     output_path="../data_apply_SMOTE/PCA/indian_liver_patient_train_k_fold_04_PCA.csv"
# )
# KFold = 5
# apply_PCA_1D(
#     input_path="../data_apply_SMOTE/KFold_data/indian_liver_patient_train_k_fold_05_SMOTE.csv",
#     output_path="../data_apply_SMOTE/PCA/indian_liver_patient_train_k_fold_05_PCA.csv"
# )
##########
##########
# Apply PCA for raw data for visualize
apply_PCA_1D(
    input_path="liver_cirrhosis_cleaned.csv",
    target_column="Stage",
    output_path="./liver_cirrhosis_cleaned_after_PCA.csv"
)
# Apply PCA for KFold data after SMOTE
# KFold = 1
apply_PCA_1D(
    input_path="../data_apply_SMOTE/KFold_data/liver_cirrhosis_train_k_fold_01_SMOTE.csv",
    target_column="Stage",
    output_path="../data_apply_SMOTE/PCA/liver_cirrhosis_train_k_fold_01_PCA.csv"
)
# KFold = 2
apply_PCA_1D(
    input_path="../data_apply_SMOTE/KFold_data/liver_cirrhosis_train_k_fold_02_SMOTE.csv",
    target_column="Stage",
    output_path="../data_apply_SMOTE/PCA/liver_cirrhosis_train_k_fold_02_PCA.csv"
)
# KFold = 3
apply_PCA_1D(
    input_path="../data_apply_SMOTE/KFold_data/liver_cirrhosis_train_k_fold_03_SMOTE.csv",
    target_column="Stage",
    output_path="../data_apply_SMOTE/PCA/liver_cirrhosis_train_k_fold_03_PCA.csv"
)
# KFold = 4
apply_PCA_1D(
    input_path="../data_apply_SMOTE/KFold_data/liver_cirrhosis_train_k_fold_04_SMOTE.csv",
    target_column="Stage",
    output_path="../data_apply_SMOTE/PCA/liver_cirrhosis_train_k_fold_04_PCA.csv"
)
# KFold = 5
apply_PCA_1D(
    input_path="../data_apply_SMOTE/KFold_data/liver_cirrhosis_train_k_fold_05_SMOTE.csv",
    target_column="Stage",
    output_path="../data_apply_SMOTE/PCA/liver_cirrhosis_train_k_fold_05_PCA.csv"
)