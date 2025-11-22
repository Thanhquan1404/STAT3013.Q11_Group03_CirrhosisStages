import pandas as pd
from sklearn.model_selection import KFold
import os

def split_kfold_save(input_path: str,
                     output_dir: str,
                     target_column: str,
                     k_folds=5,
                     random_state=42):
    """
    Split dataset into K folds and save train/test CSVs for each fold.
    """
    # Load data
    df = pd.read_csv(input_path)
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize KFold
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    
    # Loop over folds
    for fold, (train_idx, test_idx) in enumerate(kf.split(df), start=1):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        # Format fold number as 2 digits
        fold_str = str(fold).zfill(2)
        
        # File paths
        train_path = os.path.join(output_dir, f"indian_liver_patient_train_k_fold_{fold_str}.csv")
        test_path = os.path.join(output_dir, f"indian_liver_patient_test_k_fold_{fold_str}.csv")
        
        # Save CSVs
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"Fold {fold}: Train shape {train_df.shape}, Test shape {test_df.shape}")
        print(f"Saved to {train_path} and {test_path}\n")

# Split general data into 5 K_Fold
# split_kfold_save(input_path="indian_liver_patient_cleaned.csv",
#                  output_dir="../KFold_data",
#                  target_column="Result",
#                  k_folds=5,
#                  random_state=42
#                 )
