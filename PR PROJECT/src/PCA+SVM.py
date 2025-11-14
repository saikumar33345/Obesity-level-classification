
"""
PCA + SVM Classification 
-------------------------------------------------------
Steps:
1. Load one-hot encoded & already scaled X_train/X_test
2. Apply PCA (retain 95% variance)
3. Train SVM using 10-fold GridSearchCV
4. Evaluate performance
5. Save confusion matrix + PCA explained variance
6. Save results to results/pca_svm_results.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import itertools
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR = "results"


# ---------------- LOAD DATA ----------------
def load_data():
    X_train = pd.read_csv("data/X_train.csv")
    X_test  = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv").squeeze()
    y_test  = pd.read_csv("data/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


# ---------- PLOT CONFUSION MATRIX ----------
def plot_confusion_matrix(cm, classes, title, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)

    threshold = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center",
                 color="white" if cm[i, j] > threshold else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------------- MAIN TRAINING ----------------
def main():

    print("Loading encoded + scaled dataset...")
    X_train, X_test, y_train, y_test = load_data()

    # ----------------------------------------------------------
    # IMPORTANT: Your dataset is ALREADY one-hot encoded + scaled
    # So DO NOT scale again here (fixes accuracy jump issues)
    # ----------------------------------------------------------

    X_train_s = X_train.values
    X_test_s  = X_test.values

    # ---- PCA ----
    print("Running PCA...")
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca  = pca.transform(X_test_s)

    n_components = X_train_pca.shape[1]
    print(f"PCA reduced features from {X_train.shape[1]} → {n_components}")

    # PCA variance curve
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/pca_explained_variance_svm.png")
    plt.close()

    # ---- SVM with CV ----
    print("Training SVM with 10-fold CV...")

    svm = SVC()

    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": ["scale"],
        "kernel": ["rbf"]
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    grid = GridSearchCV(svm, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train_pca, y_train)

    best = grid.best_estimator_
    print("Best Params:", grid.best_params_)

    # ---- Final model ----
    best.fit(X_train_pca, y_train)
    y_pred = best.predict(X_test_pca)

    # ---- Evaluation ----
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    print("\nPerformance:")
    print("Accuracy:", acc)
    print("Macro F1:", f1m)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ---- Save confusion matrix ----
    classes = sorted(map(str, np.unique(y_test)))
    cm_path = f"{RESULTS_DIR}/pca_svm_confusion_matrix.png"
    plot_confusion_matrix(cm, classes, "PCA + SVM Confusion Matrix", cm_path)
    print("Saved confusion matrix →", cm_path)

    # ---- Save results CSV ----
    results_df = pd.DataFrame([{
        "model": "PCA + SVM",
        "pca_components": n_components,
        "accuracy": acc,
        "macro_f1": f1m
    }])
    results_df.to_csv(f"{RESULTS_DIR}/pca_svm_results.csv", index=False)
    print("Saved results → results/pca_svm_results.csv")

    print("PCA + SVM completed.")


if __name__ == "__main__":
    main()
