
"""
PCA + Logistic Regression 

- Dataset is already:
    * one-hot encoded
    * scaled
    * split properly (no leakage)
- So we do NOT scale again.

Steps:
1. Load processed X_train, X_test
2. Apply PCA on entire feature set (retain 95% variance)
3. Train Logistic Regression using 10-fold GridSearchCV
4. Save confusion matrix + results CSV
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
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


# ---------- CONFUSION MATRIX ----------
def plot_confusion_matrix(cm, classes, title, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------------- MAIN ----------------
def main():

    print("Loading preprocessed dataset...")
    X_train, X_test, y_train, y_test = load_data()

    # -------------------------------------
    # No scaling here — already scaled!
    # Use full dataset for PCA input.
    # -------------------------------------
    X_train_s = X_train.values
    X_test_s  = X_test.values

    # ---- PCA ----
    print("Running PCA...")
    pca = PCA(n_components=0.95)  # keep 95% variance
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca  = pca.transform(X_test_s)

    n_components = X_train_pca.shape[1]
    print(f"PCA reduced features from {X_train.shape[1]} → {n_components}")

    # Save PCA explained variance
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/pca_explained_variance_logreg.png")
    plt.close()

    # ---- Logistic Regression ----
    print("Training Logistic Regression with 10-fold CV...")

    model = LogisticRegression(multi_class="multinomial")

    param_grid = {
        "C": [0.1, 1, 10],
        "max_iter": [200, 300, 500],
        "solver": ["lbfgs"]
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    grid = GridSearchCV(model, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
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

    print("\nAccuracy:", acc)
    print("Macro F1:", f1m)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # ---- Confusion Matrix ----
    cm_path = f"{RESULTS_DIR}/pca_logreg_confusion_matrix.png"
    classes = sorted(map(str, np.unique(y_test)))
    plot_confusion_matrix(cm, classes, "PCA + Logistic Regression Confusion Matrix", cm_path)

    # ---- Save CSV ----
    results_df = pd.DataFrame([{
        "model": "PCA + Logistic Regression",
        "pca_components": n_components,
        "accuracy": acc,
        "macro_f1": f1m
    }])

    results_df.to_csv(f"{RESULTS_DIR}/pca_logreg_results.csv", index=False)
    print("Saved results → results/pca_logreg_results.csv")


if __name__ == "__main__":
    main()
