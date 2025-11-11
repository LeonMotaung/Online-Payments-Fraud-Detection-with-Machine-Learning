# 💳 Online Payments Fraud Detection with Machine Learning

**Author:** Leon Motaung  
**Environment:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly  

---

## 🧠 Project Overview

This project detects **fraudulent online payment transactions** using **machine learning**.  
The dataset contains **284,808 transactions** with 30 features (`V1–V28`), plus `Amount`, `Time`, and `Class` (fraud = 1, normal = 0).

After **cleaning, preprocessing, and feature engineering**, the processed dataset is saved as `creditcard_final.csv`.

---

## 📥 Data Source

The original dataset is publicly available at [Figshare: Credit Card Dataset](https://figshare.com/articles/dataset/creditcard_Dataset/29270873?file=55234994).

---

## ⚙️ Data Preprocessing

Steps performed:

- ✅ Handled missing values  
- ✅ Corrected data types  
- ✅ Encoded categorical variables (none present in this dataset)  
- ✅ Scaled features (`Amount` and `Time`)  
- ✅ Removed outliers using IQR  
- ✅ Added feature: log-transformed `scaled_amount`  
- ✅ Saved cleaned dataset as `creditcard_final.csv`  

---

## 📊 Data Visualization

The visual exploration shows **class imbalance, feature distributions, and correlations**.

### 1️⃣ Boxplots
Boxplots help detect outliers and feature spread.

![Boxplots](box.png)

### 2️⃣ Histograms & Feature Distributions
Feature distributions after scaling.

![Feature Charts](charts.png)

### 3️⃣ Correlation Heatmap
Shows relationships between features and the `Class` label.

![Heatmap](heatmap.png)

### 4️⃣ Scatterplot
Scatterplot demonstrates patterns between two features (`V2` vs `V4`).

![Scatterplot](scatterplot.png)

### 5️⃣ Class Distribution
The dataset is **highly imbalanced** — frauds are much fewer than normal transactions.

![Class Distribution](bar_chart.png)

---

## 🧩 Baseline Models Evaluation

The three baseline models were trained on the **final training set** and evaluated on the **validation set**.  

| Model                  | Accuracy     | Precision   | Recall     | F1-score   | Notes |
|------------------------|------------|------------|-----------|-----------|-------|
| Logistic Regression     | 0.9768     | 0.0577     | 0.8929    | 0.1083    | Simple, interpretable, baseline model |
| Decision Tree           | 0.9993     | 0.8478     | 0.6964    | 0.7647    | Visualize rules, understand feature importance |
| K-Nearest Neighbors (KNN)| 0.9995     | 0.9744     | 0.6786    | 0.8000    | Works on small datasets, pattern recognition |

> **Note:** The dataset is highly imbalanced, so metrics like **Recall and F1-score** are more important than Accuracy for fraud detection.

---

## 🧩 Next Steps

1. Handle **class imbalance** (SMOTE or undersampling).  
2. Train more advanced models: XGBoost, LightGBM.  
3. Evaluate using **Precision, Recall, F1-score, ROC-AUC**.  
4. Deploy the trained model via Flask or Streamlit dashboard.

---

## 📁 Project Structure

