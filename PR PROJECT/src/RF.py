
"""
Random Forest Classification (NO PCA)
-------------------------------------
Works directly on:
 - One-hot encoded categorical features
 - Scaled numeric features (from feature_engineering.py)

Pipeline:
1. Load X_train, X_test, y_train, y_test
2. Train RandomForest with 10-fold CV
3. Evaluate
4. Save confusion matrix + results CSV
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.ensemble import RandomForestClassifier
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

    print("Loading processed dataset...")
    X_train, X_test, y_train, y_test = load_data()

    # Convert to NumPy
    X_train_np = X_train.values
    X_test_np = X_test.values

    # ---- Random Forest + GridSearchCV ----
    print("Training Random Forest with 10-fold CV...")

    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train_np, y_train)

    best = grid.best_estimator_
    print("Best Params:", grid.best_params_)

    # ---- Final training ----
    best.fit(X_train_np, y_train)
    y_pred = best.predict(X_test_np)

    # ---- Evaluation ----
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    print("\nAccuracy:", acc)
    print("Macro F1:", f1m)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # ---- Confusion Matrix ----
    cm_path = f"{RESULTS_DIR}/rf_confusion_matrix.png"
    classes = sorted(map(str, np.unique(y_test)))
    plot_confusion_matrix(cm, classes, "Random Forest Confusion Matrix", cm_path)

    # ---- Save CSV ----
    results_df = pd.DataFrame([{
        "model": "Random Forest",
        "accuracy": acc,
        "macro_f1": f1m
    }])

    results_df.to_csv(f"{RESULTS_DIR}/rf_results.csv", index=False)
    print("Saved results → results/rf_results.csv")

    print("\nRandom Forest (NO PCA) completed.")


if __name__ == "__main__":
    main()
