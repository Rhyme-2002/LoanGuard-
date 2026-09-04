mport os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (OneHotEncoder, StandardScaler)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve)
from xgboost import XGBClassifier


df = pd.read_csv("C:/Users/HP/Downloads/banking_data_2000_messy.csv")

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# DATA PREVIEW
print(df.head())

# DATASET INFORMATION
print(df.info())

# MISSING VALUES
print(df.isnull().sum())

# REQUIRED COLUMNS
required_columns = ["age", "monthly_income_bdt", "account_balance_bdt", "credit_score", "loan_amount_bdt", "loan_tenure_months", "interest_rate_pct",
                    "monthly_installment_bdt", "previous_loans", "transaction_frequency_monthly", "gender", "division", "district", "education",
                    "employment_type", "account_type", "loan_type", "previous_default", "default"]

missing_columns = [col for col in required_columns
                   if col not in df.columns]

if missing_columns:
    raise ValueError(f"Dataset is missing required columns:\n"
                     f"{missing_columns}")

# REMOVE ID COLUMNS
df = df.drop(columns=["customer_id", "customer_name"], errors="ignore")

# FEATURE ENGINEERING
df["loan_to_income_ratio"] = (df["loan_amount_bdt"] / (df["monthly_income_bdt"] * 12))
df["installment_to_income_ratio"] = (df["monthly_installment_bdt"] / df["monthly_income_bdt"])

# Replace infinity values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# BASIC STATISTICS
defaulters = df[df["default"] == 1].copy()
non_defaulters = df[df["default"] == 0].copy()
total_customers = len(df)
total_defaulters = len(defaulters)
total_non_defaulters = len(non_defaulters)
default_rate = (total_defaulters / total_customers) * 100
total_defaulted_loan = (defaulters["loan_amount_bdt"].sum())
avg_defaulted_loan = (defaulters["loan_amount_bdt"].mean())
avg_defaulter_income = (defaulters["monthly_income_bdt"].mean())
avg_defaulter_credit_score = (defaulters["credit_score"].mean())
avg_defaulter_age = (defaulters["age"].mean())

# DASHBOARD STATISTICS

print("Total Customers:", total_customers)
print("Total Defaulters:", total_defaulters)
print("Total Non-Defaulters:", total_non_defaulters)
print("Default Rate:", default_rate, "%")

print("Total Defaulted Loan: BDT", total_defaulted_loan)
print("Average Defaulted Loan: BDT", avg_defaulted_loan)
print("Average Defaulter Income: BDT", avg_defaulter_income)

print("Average Defaulter Credit Score:", avg_defaulter_credit_score)
print("Average Defaulter Age:", avg_defaulter_age)


# FEATURE LISTS

numeric_features = ["age", "monthly_income_bdt", "account_balance_bdt", "credit_score", "loan_amount_bdt", "loan_tenure_months",
                    "interest_rate_pct", "monthly_installment_bdt", "previous_loans", "transaction_frequency_monthly",
                    "loan_to_income_ratio", "installment_to_income_ratio"]

categorical_features = ["gender", "division", "district", "education", "employment_type", "account_type", "loan_type", "previous_default"]

# EXPLORATORY DATA ANALYSIS

plt.figure()
sns.countplot(data=df, x="default") 
plt.xticks([0, 1], ["Non-Defaulter", "Defaulter"])
plt.title("Default Distribution")
plt.show()


# DEFAULT PERCENTAGE
counts = df["default"].value_counts().sort_index()

plt.figure()
plt.pie(counts, labels=["Non-Defaulter", "Defaulter"], autopct="%1.1f%%")
plt.title("Default Percentage")
plt.show()

# NUMERICAL DISTRIBUTION
for variable in numeric_features:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=variable, hue="default", kde=True, bins=30)
    plt.title(f"Distribution of {variable}")
    plt.show()

# DEFAULTER PROFILE ANALYSIS

for variable in categorical_features:
    default_rate_analysis = (df.groupby(variable)["default"].mean().mul(100).sort_values(ascending=False))
    print(f"Default Rate by {variable}")
    print(default_rate_analysis)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=default_rate_analysis.values, y=default_rate_analysis.index)
    plt.xlabel("Default Rate (%)")
    plt.title(f"Default Rate by {variable}")
    plt.show()

# CORRELATION ANALYSIS
correlation = df[numeric_features + ["default"]].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.show()

# RELATIONSHIP WITH DEFAULT
default_corr = correlation["default"].drop("default").sort_values(key=abs, ascending=False)

print("\nRelationship with Default")
print(default_corr)
