"""
Main Pipeline Script
--------------------
Runs the ML workflow in correct order:

1. Feature Engineering
2. PCA + SVM
3. PCA + Logistic Regression
4. Random Forest (NO PCA)
5. KBest (SVM + Logistic Regression)
6. Combine all results into final_comparison.csv
"""

import os
import pandas as pd

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# -----------------------------
# 1. Feature Engineering
# -----------------------------
print("\nSTEP 1: Running Feature Engineering...")
os.system("python src/feature_engineering.py")


# -----------------------------
# 2. PCA + SVM
# -----------------------------
print("\nSTEP 2: Running PCA + SVM...")
os.system("python src/PCA+SVM.py")


# -----------------------------
# 3. PCA + Logistic Regression
# -----------------------------
print("\nSTEP 3: Running PCA + LR...")
os.system("python src/PCA+LR.py")


# -----------------------------
# 4. Random Forest (NO PCA)
# -----------------------------
print("\nSTEP 4: Running Random Forest (no PCA)...")
os.system("python src/RF.py")


# -----------------------------
# 5. KBest (SVM + LR)
# -----------------------------
print("\nSTEP 5: Running KBest (SVM + Logistic Regression)...")
os.system("python src/Kbest(SVM+LR).py")


# -----------------------------
# 6. Combine Results
# -----------------------------
print("\nSTEP 6: Combining all results into final_comparison.csv...")

result_files = {
    "PCA + SVM": "results/pca_svm_results.csv",
    "PCA + LR": "results/pca_logreg_results.csv",
    "Random Forest": "results/rf_results.csv",
    "KBest Models": "results/kbest_results.csv"
}

all_results = []


def read_file(path, label):
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["model_group"] = label
        return df
    else:
        print(f"Warning: Missing results file → {path}")
        return None


for label, file_path in result_files.items():
    df = read_file(file_path, label)
    if df is not None:
        all_results.append(df)


final_df = pd.concat(all_results, ignore_index=True)
final_df.to_csv("results/final_comparison.csv", index=False)

print("\nFinal comparison saved → results/final_comparison.csv")
print("\nPipeline execution completed successfully.")
