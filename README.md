# 💳 Online Payments Fraud Detection with Machine Learning

**Author:** Leon Motaung  
**Company:** DeWet Technologies  
**Environment:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly  

---

## 🧠 Project Overview

This project aims to detect **fraudulent online payment transactions** using **machine learning** techniques.  
The dataset (`creditcard.csv`) contains over **284,000 transactions** with 30 anonymized features (`V1–V28`), along with `Amount`, `Time`, and `Class` (fraud or normal).

After thorough **data cleaning**, **visualization**, and **feature engineering**, the processed dataset is saved as `creditcard_final.csv`.

---

## ⚙️ Data Preprocessing

Key preprocessing steps performed:

- ✅ **Handled Missing Values** – removed or filled nulls  
- ✅ **Converted Data Types** – ensured numerical consistency  
- ✅ **Scaled Features** – normalized `Amount` and `Time` using `RobustScaler`  
- ✅ **Removed Outliers** – using IQR method  
- ✅ **Saved Clean Dataset** – exported as `creditcard_final.csv`  

```python
data.to_csv("creditcard_final.csv", index=False)
