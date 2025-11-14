"""
Proper Feature Engineering (NO DATA LEAKAGE)

Pipeline:
1. Load dataset
2. Split into Train/Test
3. Fit One-Hot Encoder on TRAIN only
4. Transform Train and Test separately
5. Fit StandardScaler on TRAIN only
6. Transform Train and Test separately
7. Save X_train, X_test, y_train, y_test

Output:
 - data/X_train.csv
 - data/X_test.csv
 - data/y_train.csv
 - data/y_test.csv
 - data/obesity_clean.csv
 - data/artifacts/
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import joblib


def prepare_and_save_clean_data():

    # ----------- Load dataset -----------
    data_path = "data/obesity.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found!")

    df = pd.read_csv(data_path)
    print("Loaded dataset:", df.shape)

    target_col = "NObeyesdad"
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # ----------- Derived Feature: BMI -----------
    if "Height" in X.columns and "Weight" in X.columns:
        X["Height_m"] = X["Height"] / 100
        X["BMI"] = X["Weight"] / (X["Height_m"] ** 2)

    # ----------- Split BEFORE encoding/scaling (IMPORTANT) -----------
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Train:", X_train_raw.shape, "Test:", X_test_raw.shape)

    # ----------- Separate column types -----------
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print("Categorical:", categorical_cols)
    print("Numeric:", numeric_cols)

    # ----------- LABEL ENCODE TARGET (fit only on train) -----------
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    os.makedirs("data/artifacts", exist_ok=True)
    joblib.dump(le, "data/artifacts/label_encoder_target.joblib")

    # ----------- One-Hot Encoding on TRAIN only -----------
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    X_train_cat = ohe.fit_transform(X_train_raw[categorical_cols])
    X_test_cat = ohe.transform(X_test_raw[categorical_cols])

    joblib.dump(ohe, "data/artifacts/onehot_encoder.joblib")

    # ----------- Keep numeric features -----------
    X_train_num = X_train_raw[numeric_cols].values
    X_test_num = X_test_raw[numeric_cols].values

    # ----------- Scale numeric (fit only on TRAIN) -----------
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)

    joblib.dump(scaler, "data/artifacts/scaler.joblib")

    # ----------- Combine numeric + one-hot encoded categorical -----------
    X_train = np.hstack([X_train_num_scaled, X_train_cat])
    X_test = np.hstack([X_test_num_scaled, X_test_cat])

    # Retrieve OHE output column names
    ohe_cols = ohe.get_feature_names_out(categorical_cols)
    final_cols = numeric_cols + ohe_cols.tolist()

    X_train_df = pd.DataFrame(X_train, columns=final_cols)
    X_test_df = pd.DataFrame(X_test, columns=final_cols)

    # ----------- Save clean datasets -----------
    os.makedirs("data", exist_ok=True)
    X_train_df.to_csv("data/X_train.csv", index=False)
    X_test_df.to_csv("data/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv("data/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("data/y_test.csv", index=False)

    # Also combine full clean dataset (optional)
    full_clean = pd.concat([
        X_train_df.assign(target=y_train),
        X_test_df.assign(target=y_test)
    ])
    full_clean.to_csv("data/obesity_clean.csv", index=False)

    print("Feature Engineering Completed Successfully!")
    print("Final Train:", X_train_df.shape, "Final Test:", X_test_df.shape)


if __name__ == "__main__":
    prepare_and_save_clean_data()
