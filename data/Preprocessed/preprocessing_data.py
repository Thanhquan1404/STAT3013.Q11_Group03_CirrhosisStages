import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_indian_liver_data(
    input_path="../Raw/indian_liver_patient_preprocessed.csv",
    output_path="indian_liver_patient_cleaned.csv"
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
df_cleaned = clean_indian_liver_data()