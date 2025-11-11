import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler # Added for scaling

def main():
    # --- 1. Load Data ---
    try:
        # Corrected filename
        data = pd.read_csv("creditcard.csv") 
    except FileNotFoundError:
        print("Error: 'creditcard.csv' not found. Please check the file path.")
        return

    # --- 2. Initial Data Inspection ---
    print("--- Data Head ---")
    print(data.head())
    
    print("\n--- Null Values Check ---")
    print(data.isnull().sum())


    # --- 3. Analyze and Visualize Fraud Class Distribution ---
    print("\n--- Analyzing Fraud Class Distribution ---")
    
    # Changed from "type" to "Class"
    class_counts = data["Class"].value_counts()
    
    # Map index 0/1 to meaningful names
    class_names = class_counts.index.map({0: 'Non-Fraud', 1: 'Fraud'})
    quantity = class_counts.values

    # Create and show the pie chart
    figure = px.pie(values=quantity, 
                     # Use the new mapped names
                     names=class_names, 
                     hole=0.5, 
                     title="Distribution of Fraud vs. Non-Fraud Transactions")
    figure.show()


    # --- 4. Correlation Analysis ---
    print("\n--- Correlation with 'Class' (Fraud) ---")
    try:
        correlation = data.corr(numeric_only=True)
        # Changed from "isFraud" to "Class"
        print(correlation["Class"].sort_values(ascending=False))
    except KeyError as e:
        print(f"Error calculating correlation: {e}. Check if columns are numeric.")
    except Exception as e:
        print(f"An error occurred during correlation: {e}")


    # --- 5. Feature Scaling (Preparation for Modeling) ---
    # The V1-V28 columns are already scaled (from PCA).
    # We must scale 'Time' and 'Amount' so they are on the same range.
    
    print("\n--- Scaling 'Time' and 'Amount' features ---")
    scaler = StandardScaler()
    
    # Scale 'Amount' and 'Time'
    data['scaled_Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    data['scaled_Time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))
    
    # Drop the original unscaled columns
    data = data.drop(['Time', 'Amount'], axis=1)
    
    print("\n--- Data Head After Scaling ---")
    # Re-order columns to put 'Class' at the end (conventional)
    cols = [col for col in data.columns if col != 'Class'] + ['Class']
    data = data[cols]
    print(data.head())


# Standard Python entry point
if __name__ == "__main__":
    main()
