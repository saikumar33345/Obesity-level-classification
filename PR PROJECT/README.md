# Obesity Classification Using PCA, KBest & Classical ML Models Based on eating habits 
A complete machine learning pipeline for predicting obesity levels using PCA, SelectKBest, Logistic Regression, SVM, and Random Forest.

---

## Project Overview

This project builds a **full end-to-end ML classification pipeline**:

1. **Proper Feature Engineering** (NO data leakage)
2. **One-Hot Encoding** on categorical variables
3. **Standard Scaling** on numeric variables
4. **Dimensionality Reduction**
   - PCA (Logistic Regression, SVM, Random Forest)
   - SelectKBest (LR + SVM)
5. **10-Fold Cross Validation**
6. **Confusion Matrices & Results CSVs**
7. **Final comparison between all models**

---

##  Project Structure

```
PR PROJECT/
│
├── data/
│   ├── obesity.csv
│   ├── obesity_clean.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   ├── X_train_selectk.csv
│   ├── X_test_selectk.csv
│   └── artifacts/
│       ├── onehot_encoder.joblib
│       ├── scaler.joblib
│       └── label_encoder_target.joblib
│
├── results/
│   ├── pca_svm_results.csv
│   ├── pca_logreg_results.csv
│   ├── pca_random_forest_results.csv
│   ├── kbest_results.csv
│   ├── final_comparison.csv
│   ├── Confusion Matrices (PNG)
│   └── PCA Explained Variance Plots
│
├── src/
│   ├── feature_engineering.py
│   ├── PCA+SVM.py
│   ├── PCA+LR.py
│   ├── RF.py
│   ├── Kbest(SVM+LR).py
│   └── main.py
│
├── requirements.txt
└── README.md
```

---

##  How to Run the Project

### Step 1 — Create Virtual Environment
```sh
python -m venv venv
venv\Scripts\activate  (Windows)
source venv/bin/activate (Linux/Mac)
```

### Step 2 — Install Requirements
```sh
pip install -r requirements.txt
```

---

## 📌 Step 3 — Run Feature Engineering (Must Run First)
```sh
python src/feature_engineering.py
```

This generates:
- X_train.csv  
- X_test.csv  
- y_train.csv  
- y_test.csv  
- one-hot encoder  
- scaler  
- label encoder  
- obesity_clean.csv  

---

## 📌 Step 4 — Run ML Models

### PCA + SVM
```sh
python src/PCA+SVM.py
```

### PCA + Logistic Regression
```sh
python src/PCA+LR.py
```

### PCA + Random Forest
```sh
python src/RF.py
```

### SelectKBest (SVM + LR)
```sh
python src/"Kbest(SVM+LR).py"
```

All results are saved in:

```
results/
```

---

##  Model Performance Summary

| Model | Accuracy | Macro F1 |
|-------|----------|-----------|
| PCA + SVM | ~96% | 
| PCA + Logistic Regression | ~97% |
| Random Forest | ~98% |
| KBest + SVM | ~97% |
| KBest + LR | ~96% |

---

## ✔ Features of This Project

- NO data leakage  
- Clean train/test split  
- One-Hot Encoding only on training data  
- Scaling only on training data  
- PCA applied correctly  
- 10-fold cross validation  
- Automatic results saving  
- Confusion matrix heatmaps  
- Ranked feature importance (SelectKBest)

---

---
