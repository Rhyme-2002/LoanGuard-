import os
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


# PREPROCESSING
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                                      ("scaler", StandardScaler())])

# CATEGORICAL TRANSFORMATION
categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                          ("encoder", OneHotEncoder(handle_unknown="ignore"))])


# COMBINE PREPROCESSING
preprocessor = ColumnTransformer(transformers=[("numeric", numeric_transformer, numeric_features),
                                               ("categorical", categorical_transformer, categorical_features)])

# FEATURES AND TARGET
X = df[numeric_features + categorical_features].copy()
y = df["default"].copy()

# CHECK TARGET
if y.nunique() < 2:
    raise ValueError("The 'default' column must contain " 
                     "both 0 and 1 classes.")

# TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print(f"Training Samples: {X_train.shape[0]}")
print(f"Testing Samples: {X_test.shape[0]}")

# MACHINE LEARNING MODELS
models = {"Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
          "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_split=10, min_samples_leaf=5, class_weight="balanced", random_state=42),
          "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=10, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1),
          "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42),
          "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, min_child_weight=3,  subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1),
          "SVM": SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=42),
          "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance")
}

# TRAIN MODELS

trained_models = {}
results = []
roc_data = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # TRAIN MODEL
    pipeline.fit(X_train, y_train)

    # STORE MODEL
    trained_models[name] = pipeline

    # PREDICTION
    y_pred = pipeline.predict(X_test)


    # PREDICT PROBABILITY
    y_prob = pipeline.predict_proba( X_test)[:, 1]


    # MODEL METRICS
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)


    # SAVE RESULTS
    results.append({"Model": name, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1, "ROC-AUC": auc})

    # ROC DATA
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": auc}

# MODEL RESULTS
results_df = (pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True))
print(results_df.round(4))


# SAVE MODEL RESULTS
results_df.to_csv("model_comparison.csv", index=False)

# MODEL PERFORMANCE PLOT
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="ROC-AUC", y="Model")
plt.xlim(0, 1)
plt.title("Model Performance - ROC AUC")
plt.show()

# F1 SCORE PLOT
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="F1 Score", y="Model")
plt.xlim(0, 1)
plt.title("Model Performance - F1 Score")
plt.show()

# ROC CURVE COMPARISON
plt.figure(figsize=(10, 7))

for name, data in roc_data.items():
    plt.plot(data["fpr"], data["tpr"], label=(f"{name} " f"(AUC = {data['auc']:.3f})"))

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# BEST MODEL
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
best_pred = best_model.predict(X_test)
best_prob = best_model.predict_proba(X_test)[:, 1]

# BEST MODEL PERFORMANCE
print(f"Selected Model: {best_model_name}")

accuracy = accuracy_score(y_test, best_pred)
precision = precision_score(y_test, best_pred, zero_division=0)
recall = recall_score(y_test, best_pred, zero_division=0)
f1 = f1_score(y_test, best_pred, zero_division=0)
auc = roc_auc_score(y_test, best_prob)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")

# CONFUSION MATRIX
cm = confusion_matrix(y_test, best_pred)
print(cm)


plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Defaulter", "Defaulter"], yticklabels=["Non-Defaulter", "Defaulter"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# CLASSIFICATION REPORT
report = classification_report(y_test, best_pred, target_names=["Non-Defaulter", "Defaulter"], zero_division=0)
print(report)

# FEATURE IMPORTANCE
model_object = best_model.named_steps["model"]
preprocessor_object = best_model.named_steps["preprocessor"]

if hasattr(model_object, "feature_importances_"):
    feature_names = preprocessor_object.get_feature_names_out()
    feature_importance = pd.DataFrame({"Feature": feature_names, "Importance": model_object.feature_importances_}).sort_values("Importance", ascending=False).head(20)

    print(feature_importance)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance, x="Importance", y="Feature")
    plt.title("Top 20 Feature Importance")
    plt.show()

print("\nEnter customer information:")

# NUMERICAL INFORMATION
age = int(input("\nAge: "))
monthly_income_bdt = float(input("Monthly Income (BDT): "))
account_balance_bdt = float(input("Account Balance (BDT): "))
credit_score = float(input("Credit Score: "))
loan_amount_bdt = float(input("Loan Amount (BDT): "))
loan_tenure_months = int(input("Loan Tenure (Months): "))
interest_rate_pct = float(input("Interest Rate (%): "))
monthly_installment_bdt = float(input("Monthly Installment (BDT): "))
previous_loans = int(input("Previous Loans: "))
transaction_frequency_monthly = int(input("Monthly Transaction Frequency: "))

# CATEGORICAL INFORMATION
print("\nAvailable Gender:")
print(sorted(df["gender"].dropna().astype(str).unique()))

gender = input("Gender: ")

print("\nAvailable Division:")
print(sorted(df["division"].dropna().astype(str).unique()))

division = input("Division: ")

print("\nAvailable District:")
print(sorted(df["district"].dropna().astype(str).unique()))

district = input("District: ")

print("\nAvailable Education:")
print(sorted(df["education"].dropna().astype(str).unique()))

education = input("Education: ")

print("\nAvailable Employment Type:")
print(sorted(df["employment_type"].dropna().astype(str).unique()))

employment_type = input("Employment Type: ")

print("\nAvailable Account Type:")
print(sorted(df["account_type"].dropna().astype(str).unique()))

account_type = input("Account Type: ")

print("\nAvailable Loan Type:")
print(sorted(df["loan_type"].dropna().astype(str).unique()))

loan_type = input("Loan Type: ")

print("\nAvailable Previous Default:")
print(sorted(df["previous_default"].dropna().astype(str).unique()))

previous_default = input("Previous Default: ")

# CREATE NEW CUSTOMER DATA
new_customer_df = pd.DataFrame({"age": [age],
                                "monthly_income_bdt": [monthly_income_bdt],
                                "account_balance_bdt": [account_balance_bdt],
                                "credit_score": [credit_score],
                                "loan_amount_bdt": [loan_amount_bdt],
                                "loan_tenure_months": [loan_tenure_months],
                                "interest_rate_pct": [interest_rate_pct],
                                "monthly_installment_bdt": [monthly_installment_bdt],
                                "previous_loans": [previous_loans],
                                "transaction_frequency_monthly": [transaction_frequency_monthly],
                                "loan_to_income_ratio": [loan_amount_bdt / (monthly_income_bdt * 12)],
                                "installment_to_income_ratio": [monthly_installment_bdt / monthly_income_bdt],
                                "gender": [gender],
                                "division": [division],
                                "district": [district],
                                "education": [education],
                                "employment_type": [employment_type],
                                "account_type": [account_type],
                                "loan_type": [loan_type],
                                "previous_default": [previous_default]})


# PREDICTION
new_prediction = best_model.predict(new_customer_df)[0]
new_probability = best_model.predict_proba(new_customer_df)[0, 1]


# RISK LEVEL
if new_probability >= 0.70:
    risk_level = ("HIGH RISK")
elif new_probability >= 0.40:
    risk_level = ("MEDIUM RISK")
else:
    risk_level = ("LOW RISK")

# PREDICTION LABEL
prediction_label = "Likely Defaulter" if new_prediction == 1 else "Likely Non-Defaulter"

# FINAL RESULT
print(f"\nBest Model: {best_model_name}")
print(f"Prediction: {prediction_label}")
print(f"Default Probability: {new_probability:.2%}")
print(f"Risk Level: {risk_level}")


# CREATE PREDICTION REPORT
report_df = pd.DataFrame({"Best_Model": [best_model_name],
                          "Prediction":[prediction_label],
                          "Default_Probability": [new_probability],
                          "Risk_Level": [risk_level]})

print(report_df)

# SAVE CUSTOMER REPORT

report_df.to_csv("customer_risk_prediction.csv", index=False)
