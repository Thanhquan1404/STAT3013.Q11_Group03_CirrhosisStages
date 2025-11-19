# experiment/softmax_regression_experiment.py
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    Normalizer, MaxAbsScaler, QuantileTransformer, LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

from src.softmax_regression import SoftmaxRegressionGD


# ============================================================
# Scaler Factory
# ============================================================
def get_scaler(name: str):
    scalers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
        "Normalizer": Normalizer(),
        "MaxAbsScaler": MaxAbsScaler(),
        "QuantileTransformer": QuantileTransformer(
            output_distribution="normal", random_state=42
        ),
        None: None
    }
    if name not in scalers:
        raise ValueError(f"Unsupported scaler: {name}")
    return scalers[name]


# ============================================================
# UNIVERSAL PREPROCESSING
# ============================================================
def universal_preprocessing(path, scaler_name="StandardScaler", apply_smote=True, random_state=42):

    df = pd.read_csv(path)

    possible_labels = ["Result", "Dataset", "Class", "selector", "target", "Stage", "status", "Diagnosis"]
    label_col = next((c for c in possible_labels if c in df.columns), None)
    if label_col is None:
        raise ValueError("Label column not found.")

    y_raw = df[label_col].copy()

    unique_vals = np.unique(y_raw.dropna())

    # Determine task type
    if len(unique_vals) > 2 or label_col == "Stage":
        task = "multiclass"
        # Create label encoder → ensure labels become 0..K-1
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        original_labels = le.classes_
    else:
        task = "binary"
        y = (y_raw == y_raw.max()).astype(int).values
        original_labels = [0, 1]

    # Drop missing labels
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    X = df.drop(columns=[label_col])

    # Encode categorical features
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Impute numeric
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = SimpleImputer(strategy="mean").fit_transform(X[num_cols])

    # Scale
    scaler = get_scaler(scaler_name)
    if scaler is not None:
        X = scaler.fit_transform(X)

    X = np.asarray(X, float)
    y = np.asarray(y, int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # SMOTE only for binary
    if apply_smote and task == "binary":
        X_train, y_train = SMOTE(random_state=random_state).fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, task, original_labels


# DATASET_PATH = "../data/processed/indian_liver_patient_preprocessed.csv"
# OUTPUT_CSV = "../experiment_result/logistic_regression_indian_liver_patient_result.csv"
DATASET_PATH = "../data/processed/liver_cirrhosis_preprocessed.csv"
OUTPUT_CSV = "../experiment_result/softmax_regression_liver_cirrhosis_result.csv"

os.makedirs("../experiment_result", exist_ok=True)

if not os.path.exists(OUTPUT_CSV):
    header = (
        "Dataset,Task,Eta,Epochs,Threshold,Test_Accuracy,Test_F1_Weighted,Test_F1_Macro,"
        "Train_Time_s,Test_Time_s,Final_Loss,Converged_Epoch,Scaler,Timestamp\n"
    )
    with open(OUTPUT_CSV, "w") as f:
        f.write(header)

CONFIGS = [
    {"eta": 0.01, "epochs": 2000, "threshold": 1e-3},
    {"eta": 0.05, "epochs": 1500, "threshold": 1e-4},
    {"eta": 0.1,  "epochs": 1000, "threshold": 1e-4},
    {"eta": 0.2,  "epochs": 800,  "threshold": 1e-4},
]

SCALERS = [None, "StandardScaler", "MinMaxScaler", "RobustScaler",
           "Normalizer", "MaxAbsScaler", "QuantileTransformer"]


# ============================================================
# Experiment Loop
# ============================================================
for scaler_name in SCALERS:

    X_train, X_test, y_train, y_test, task, original_labels = universal_preprocessing(
        DATASET_PATH, scaler_name=scaler_name, apply_smote=False
    )

    for cfg in CONFIGS:

        model = SoftmaxRegressionGD(
            eta=cfg["eta"], epochs=cfg["epochs"],
            threshold=cfg["threshold"], verbose=False
        )

        # Train
        t0 = time.time()
        model.fit(X_train, y_train, X_test, y_test)
        train_time = time.time() - t0

        # Predict
        t1 = time.time()
        y_pred = model.predict(X_test)
        test_time = time.time() - t1

        # Decode labels back to original
        if task == "multiclass":
            le_map = {i: original_labels[i] for i in range(len(original_labels))}
            y_test_eval = np.array([le_map[i] for i in y_test])
            y_pred_eval = np.array([le_map[i] for i in y_pred])
        else:
            y_test_eval = y_test
            y_pred_eval = y_pred

        # Metrics
        report = classification_report(y_test_eval, y_pred_eval, output_dict=True, zero_division=0)
        acc = report["accuracy"]
        f1w = report["weighted avg"]["f1-score"]
        f1m = report["macro avg"]["f1-score"]

        final_loss = model.loss_history[-1]
        converged_epoch = len(model.loss_history)

        # Save results
        row = [
            os.path.basename(DATASET_PATH), task,
            cfg["eta"], cfg["epochs"], cfg["threshold"],
            round(acc, 4), round(f1w, 4), round(f1m, 4),
            round(train_time, 4), round(test_time, 4),
            round(final_loss, 6), converged_epoch,
            scaler_name, time.strftime("%Y%m%d_%H%M%S")
        ]

        pd.DataFrame([row]).to_csv(OUTPUT_CSV, mode="a", index=False, header=False)

print("\nExperiment completed successfully.")
print(f"Results saved to: {OUTPUT_CSV}")
