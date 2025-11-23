import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import os

def calculate_statistics(input_path="../Data/Preprocessed/indian_liver_patient_cleaned.csv",
                         output_path="../calculation_result/statistical_calculation_result.csv"):
    """
    Calculate descriptive statistics for numeric columns and save to CSV.
    """
    # Load data
    df = pd.read_csv(input_path)
    
    # Create output directory if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    stats_list = []
    
    for col in numeric_cols:
        series = df[col]
        Q1 = series.quantile(0.25)
        Q2 = series.quantile(0.50)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        Upper_bound = Q3 + 1.5 * IQR
        Lower_bound = Q1 - 1.5 * IQR
        stats_list.append({
            "Feature": col,
            "Count": series.count(),
            "Mean": series.mean(),
            "Std": series.std(),
            "Variance": series.var(),
            "Min": series.min(),
            "Q1": Q1,
            "Median(Q2)": Q2,
            "Q3": Q3,
            "IQR": IQR,
            "Lower bound": Lower_bound,
            "Upper bound": Upper_bound,
            "Max": series.max(),
            "Range": series.max() - series.min(),
            "Skewness": skew(series),
            "Kurtosis": kurtosis(series)
        })
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(stats_list)
    
    # Save to CSV
    stats_df.to_csv(output_path, index=False)
    print(f"Statistical calculation result saved to {output_path}")
    
    return stats_df

# Usage
calculate_statistics(
    input_path="../Data/Preprocessed/indian_liver_patient_cleaned.csv",
    output_path="../calculation_result/indian_liver_patient_statistical_calculation_result.csv"
)
calculate_statistics(
    input_path="../Data/Preprocessed/liver_cirrhosis_cleaned.csv",
    output_path="../calculation_result/liver_cirhosis_statistical_calculation_result.csv"
)
