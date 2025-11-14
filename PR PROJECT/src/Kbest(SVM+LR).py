
"""
Classification on SelectKBest Reduced Features

Models:
- SVM
- Logistic Regression

This script:
1. Loads X_train_selectk.csv and X_test_selectk.csv
2. Runs 10-fold GridSearchCV
3. Saves confusion matrices + results CSV
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import warnings
warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

RESULTS_DIR = "results"


# -------------------- LOAD KBEST DATA --------------------
def load_kbest_data():
    X_train = pd.read_csv("data/X_train_selectk.csv").values
    X_test  = pd.read_csv("data/X_test_selectk.csv").values
    y_train = pd.read_csv("data/y_train.csv").squeeze()
    y_test  = pd.read_csv("data/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


# ---------------- CONFUSION MATRIX PLOT ------------------
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


# ------------------ TRAIN + EVALUATE ---------------------
def run_model(name, model, X_train, y_train, X_test, y_test, cv=10):
    print(f"\n===== {name} =====")

    if isinstance(model, SVC):
        param_grid = {"C": [0.1, 1, 10], "gamma": ["scale"], "kernel": ["rbf"]}

    elif isinstance(model, LogisticRegression):
        param_grid = {"C": [0.1, 1, 10], "max_iter": [200, 300], "solver": ["lbfgs"]}

    cvk = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    search = GridSearchCV(model, param_grid, cv=cvk, n_jobs=-1, scoring="f1_macro")
    search.fit(X_train, y_train)

    best = search.best_estimator_
    print("Best Parameters:", search.best_params_)

    best.fit(X_train, y_train)
    y_pred = best.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1m  = f1_score(y_test, y_pred, average="macro")
    cm   = confusion_matrix(y_test, y_pred)

    print(classification_report(y_test, y_pred))

    return acc, f1m, cm


# --------------------------- MAIN ------------------------
def main():
    print("Loading SelectKBest reduced data...")
    X_train, X_test, y_train, y_test = load_kbest_data()

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    models = [
        ("SVM", SVC()),
        ("LogisticRegression", LogisticRegression(multi_class="multinomial"))
    ]

    results = []

    for name, mdl in models:
        acc, f1m, cm = run_model(name, mdl, X_train, y_train, X_test, y_test)

        cm_path = os.path.join(RESULTS_DIR, f"cm_kbest_{name}.png")
        plot_confusion_matrix(cm, sorted(map(str, np.unique(y_test))), name, cm_path)

        results.append({"model": name, "accuracy": acc, "macro_f1": f1m})

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "kbest_results.csv"), index=False)

    print("\nSaved results to results/kbest_results.csv")
    print("Done.")


if __name__ == "__main__":
    main()
