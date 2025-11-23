import pandas as pd
from imblearn.over_sampling import SMOTE

def apply_SMOTE(inputPath: str, targetColumn: str, outputPath: str): 
    # Load data
    df = pd.read_csv(inputPath)
    
    # Split features and target
    X = df.drop(columns=[targetColumn])
    y = df[targetColumn]

    # --- Print class distribution before SMOTE ---
    print("Class distribution before SMOTE:")
    print(y.value_counts())
    
    # --- Apply SMOTE ---
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # --- Print class distribution after SMOTE ---
    print("Class distribution after SMOTE:")
    print(pd.Series(y_res).value_counts())
    
    # --- Combine X_res and y_res into one DataFrame ---
    output = pd.DataFrame(X_res, columns=X.columns)
    output[targetColumn] = y_res  # thêm cột target
    
    # --- Save to CSV ---
    output.to_csv(outputPath, index=False)
    
    return output

# Apply SMOTE for raw data 
# apply_SMOTE(
#     inputPath="indian_liver_patient_cleaned.csv", 
#     targetColumn="Result", 
#     outputPath="../Validate_data/indian_liver_patient_after_SMOTE.csv"
# )

# Apply SMOTE for KFold data
# KFold = 1
# apply_SMOTE(
#     inputPath="../KFold_data/indian_liver_patient_train_k_fold_01.csv",
#     targetColumn="Result",
#     outputPath="../data_apply_SMOTE/KFold_data/indian_liver_patient_train_k_fold_01_SMOTE.csv"
# )
# KFold = 2
# apply_SMOTE(
#     inputPath="../KFold_data/indian_liver_patient_train_k_fold_02.csv",
#     targetColumn="Result",
#     outputPath="../data_apply_SMOTE/KFold_data/indian_liver_patient_train_k_fold_02_SMOTE.csv"
# )
# # KFold = 3
# apply_SMOTE(
#     inputPath="../KFold_data/indian_liver_patient_train_k_fold_03.csv",
#     targetColumn="Result",
#     outputPath="../data_apply_SMOTE/KFold_data/indian_liver_patient_train_k_fold_03_SMOTE.csv"
# )
# # KFold = 4
# apply_SMOTE(
#     inputPath="../KFold_data/indian_liver_patient_train_k_fold_04.csv",
#     targetColumn="Result",
#     outputPath="../data_apply_SMOTE/KFold_data/indian_liver_patient_train_k_fold_04_SMOTE.csv"
# )
# # KFold = 5
# apply_SMOTE(
#     inputPath="../KFold_data/indian_liver_patient_train_k_fold_05.csv",
#     targetColumn="Result",
#     outputPath="../data_apply_SMOTE/KFold_data/indian_liver_patient_train_k_fold_05_SMOTE.csv"
# )
########### 
###########
# Apply SMOTE for raw data 
# apply_SMOTE(
#     inputPath="liver_cirrhosis_cleaned.csv", 
#     targetColumn="Stage", 
#     outputPath="../data_apply_SMOTE/liver_cirrhosis_after_SMOTE.csv"
# )

# Apply SMOTE for KFold data
# KFold = 1
# apply_SMOTE(
#     inputPath="../KFold_data/liver_cirrhosis_train_k_fold_01.csv",
#     targetColumn="Stage",
#     outputPath="../data_apply_SMOTE/KFold_data/liver_cirrhosis_train_k_fold_01_SMOTE.csv"
# )
# KFold = 2
# apply_SMOTE(
#     inputPath="../KFold_data/liver_cirrhosis_train_k_fold_02.csv",
#     targetColumn="Stage",
#     outputPath="../data_apply_SMOTE/KFold_data/liver_cirrhosis_train_k_fold_02_SMOTE.csv"
# )
# # KFold = 3
# apply_SMOTE(
#     inputPath="../KFold_data/liver_cirrhosis_train_k_fold_03.csv",
#     targetColumn="Stage",
#     outputPath="../data_apply_SMOTE/KFold_data/liver_cirrhosis_train_k_fold_03_SMOTE.csv"
# )
# # KFold = 4
# apply_SMOTE(
#     inputPath="../KFold_data/liver_cirrhosis_train_k_fold_04.csv",
#     targetColumn="Stage",
#     outputPath="../data_apply_SMOTE/KFold_data/liver_cirrhosis_train_k_fold_04_SMOTE.csv"
# )
# # KFold = 5
# apply_SMOTE(
#     inputPath="../KFold_data/liver_cirrhosis_train_k_fold_05.csv",
#     targetColumn="Stage",
#     outputPath="../data_apply_SMOTE/KFold_data/liver_cirrhosis_train_k_fold_05_SMOTE.csv"
# )