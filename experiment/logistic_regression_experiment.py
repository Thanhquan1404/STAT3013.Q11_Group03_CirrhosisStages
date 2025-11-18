# Experiment Pipeline for Evaluating Custom Logistic Regression (GD Variant)
# ============================================================================
# This script integrates full preprocessing (duplicate removal, imputation,
# categorical encoding, SMOTE balancing, feature scaling), model training using
# gradient descent, comprehensive evaluation, and result logging.
# Compatible with custom LogisticRegressionGD implementation.

import os
import time
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    Normalizer, QuantileTransformer, MaxAbsScaler,
    LabelEncoder, OneHotEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    matthews_corrcoef, f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, average_precision_score
)
from imblearn.over_sampling import SMOTE
from src.logistic_regression import LogisticRegressionGD


# ================================================================
# Scaler Factory
# ================================================================
def get_scaler(name: str):
    scalers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
        "Normalizer": Normalizer(),
        "QuantileTransformer": QuantileTransformer(output_distribution="normal", random_state=42),
        "MaxAbsScaler": MaxAbsScaler(),
        None: None
    }
    scaler = scalers.get(name)
    if scaler is None and name is not None:
        raise ValueError(f"Scaler {name} is not supported.")
    return scaler


def preprocessing_data(
    path: str,
    label: str = "Result",
    fillNaNValue: str = "mean",
    scaler: str = "StandardScaler",
    removeDuplicate: bool = True,
    applySMOTE: bool = True,
    test_size: float = 0.2,
    random_state: int = 42
):
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"Initial shape: {df.shape}")

    # Auto-detect label column if missing
    if label not in df.columns:
        possible_labels = ["Result", "Dataset", "Class", "selector", "target", "Label", "diagnosis"]
        for col in possible_labels:
            if col in df.columns:
                label = col
                print(f"Detected label column: {label}")
                break
        else:
            raise ValueError(f"Label column not found. Available columns: {list(df.columns)}")

    # Normalize label values to {0,1}
    raw_label = df[label].copy()
    unique_vals = sorted(raw_label.dropna().unique())

    if len(unique_vals) == 2:
        if set(unique_vals) == {1, 2}:
            df[label] = df[label].map({1: 0, 2: 1})
        elif set(unique_vals) == {0, 1}:
            df[label] = df[label].astype(int)
        else:
            df[label] = (df[label] == max(unique_vals)).astype(int)
    else:
        raise ValueError(f"Label contains more than two classes: {unique_vals}")

    # Remove rows with NaN labels
    df = df.dropna(subset=[label]).reset_index(drop=True)

    # Remove duplicates
    if removeDuplicate:
        df = df.drop_duplicates().reset_index(drop=True)

    y = df[label].astype(int)
    X = df.drop(columns=[label])

    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Impute missing values
    if num_cols:
        X[num_cols] = SimpleImputer(strategy=fillNaNValue).fit_transform(X[num_cols])
    if cat_cols:
        X[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[cat_cols])

    # Categorical encoding
    binary_cols = [c for c in cat_cols if X[c].nunique() <= 2]
    multi_cols = [c for c in cat_cols if X[c].nunique() > 2]

    for c in binary_cols:
        X[c] = LabelEncoder().fit_transform(X[c])

    if multi_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe_data = ohe.fit_transform(X[multi_cols])
        ohe_cols = ohe.get_feature_names_out(multi_cols)
        X = X.drop(columns=multi_cols)
        X[ohe_cols] = ohe_data

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # SMOTE balancing
    if applySMOTE:
        X_train, y_train = SMOTE(random_state=random_state).fit_resample(X_train, y_train)

    # Feature scaling
    scaler_obj = get_scaler(scaler)
    if scaler_obj:
        X_train = scaler_obj.fit_transform(X_train)
        X_test = scaler_obj.transform(X_test)

    # Convert to numpy
    return (
        np.asarray(X_train, dtype=np.float64),
        np.asarray(y_train, dtype=int),
        np.asarray(X_test, dtype=np.float64),
        np.asarray(y_test, dtype=int),
    )


def evaluate_model(y_true, y_pred, y_scores):
    y_true = np.ravel(y_true).astype(int)
    y_pred = np.ravel(y_pred).astype(int)
    y_scores = np.ravel(y_scores)

    auc_roc = roc_auc_score(y_true, y_scores) * 100
    auc_pr = average_precision_score(y_true, y_scores) * 100
    acc = accuracy_score(y_true, y_pred) * 100
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

    print("Evaluation completed.")
    return [auc_roc, auc_pr, acc, mcc, f1, precision, recall]


if __name__ == "__main__":
    DATASET_PATH = "../data/processed/liver_cleaned.csv"
    OUTPUT_CSV = "../experiment_result/logistic_regression_experiment_result.csv"

    PREPROCESSING_CONFIG = {
        "fillNaNValue": "mean",
        "label": "Result",
        "scaler": "RobustScaler",
        "removeDuplicate": True,
        "applySMOTE": True,
    }

    MODEL_CONFIG = {
        "eta": 0.8,
        "epochs": 2000,
        "threshold": 1e-4,
        "verbose": True,
    }

    columns = [
        "Dataset", "Model", "AUCROC", "AUCPR", "Accuracy",
        "MCC", "F1 Score", "Precision", "Recall",
        "Time Train", "Time Test"
    ]
    if not os.path.exists(OUTPUT_CSV):
        pd.DataFrame(columns=columns).to_csv(OUTPUT_CSV, index=False)

    print("Starting experiment with Logistic Regression GD...")

    # Data preprocessing
    X_train, y_train, X_test, y_test = preprocessing_data(
        path=DATASET_PATH,
        **PREPROCESSING_CONFIG,
    )

    # Model initialization
    model = LogisticRegressionGD(**MODEL_CONFIG)

    # Training
    t0 = time.time()
    model.fit(X_train, y_train)
    t1 = time.time()
    train_time = t1 - t0

    # Prediction
    y_scores = model.predict_proba(X_test).flatten()
    y_pred = model.predict(X_test).flatten()
    t2 = time.time()
    test_time = t2 - t1

    # Evaluation
    metrics = evaluate_model(y_test, y_pred, y_scores)

    # Save results
    result_row = [
        os.path.basename(DATASET_PATH),
        "LogisticRegressionGD",
        *metrics,
        train_time,
        test_time
    ]

    pd.DataFrame([result_row], columns=columns).to_csv(
        OUTPUT_CSV, mode='a', header=False, index=False
    )