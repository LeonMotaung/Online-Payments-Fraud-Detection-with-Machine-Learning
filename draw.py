import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# 1️⃣ Load cleaned dataset
# ======================
data = pd.read_csv("creditcard_final.csv")
print("Dataset loaded successfully.")
print("Shape:", data.shape)
print(data.head())

# ======================
# 2️⃣ Class Distribution
# ======================
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=data)
plt.title("Class Distribution (0 = Non-Fraud, 1 = Fraud)")
plt.xlabel("Transaction Class")
plt.ylabel("Count")
plt.show()

fraud_count = data['Class'].value_counts()
print("\nClass counts:\n", fraud_count)
print("\nPercentage of Fraud Transactions: {:.4f}%".format((fraud_count[1] / fraud_count.sum()) * 100))

# ======================
# 3️⃣ Histograms of Key Features
# ======================
features_to_plot = ['V1', 'V2', 'V3', 'V4', 'V5', 'Amount_Scaled', 'Time_Scaled']

data[features_to_plot].hist(figsize=(12, 10), bins=30, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# ======================
# 4️⃣ Boxplots to Check Outliers
# ======================
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[features_to_plot])
plt.title("Boxplots of Selected Features")
plt.xticks(rotation=45)
plt.show()

# ======================
# 5️⃣ Scatterplot: Correlation Between Two Features
# ======================
plt.figure(figsize=(6, 6))
sns.scatterplot(
    data=data.sample(5000),  # sample to reduce plot clutter
    x='V2', y='V4', hue='Class', alpha=0.6, palette='coolwarm'
)
plt.title("Scatterplot of V2 vs V4 (Colored by Class)")
plt.show()

# ======================
# 6️⃣ Correlation Heatmap
# ======================
plt.figure(figsize=(16, 12))
corr = data.corr()
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# ======================
# 7️⃣ Most Correlated Features with Fraud
# ======================
correlation_with_class = corr['Class'].sort_values(ascending=False)
print("\n🔍 Top features correlated with Fraud:")
print(correlation_with_class.head(10))
print("\nLeast correlated features:")
print(correlation_with_class.tail(10))
