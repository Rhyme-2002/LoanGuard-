import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

# PAGE CONFIGURATION
st.set_page_config(
    page_title="LoanGuard | Credit Risk Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded")

# CUSTOM CSS

st.markdown(
    """
    <style>

    /* Main page */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0;
    }

    .sub-title {
        font-size: 17px;
        text-align: center;
        color: #6b7280;
        margin-bottom: 30px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }

    /* Buttons */
    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        min-height: 45px;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 14px;
        font-weight: 600;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        border-radius: 10px;
    }

    /* Risk cards */
    .risk-high {
        background-color: #fee2e2;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }

    .risk-medium {
        background-color: #fef3c7;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: #92400e;
        border: 1px solid #fcd34d;
    }

    .risk-low {
        background-color: #dcfce7;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: #166534;
        border: 1px solid #86efac;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# APP TITLE

st.markdown(
    """
    <div class="main-title">
        🏦 LoanGuard
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
       Machine Learning based Credit Risk Analytics & Default Prediction System
    </div>
    """,
    unsafe_allow_html=True)

# SIDEBAR HEADER

st.sidebar.markdown(
    """
    <div style="text-align:center; padding:10px 5px 15px 5px;">
        <h1 style="margin-bottom:0;">🏦 LoanGuard</h1>
        <p style="color:gray; font-size:13px;">
            Credit Risk Intelligence Platform
        </p>
    </div>
    """,
    unsafe_allow_html=True)

# APP FEATURES

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 App Features")
st.sidebar.markdown(
    """
    📊 **Interactive Dashboard**
    
    🔍 **Exploratory Data Analysis**
    
    👥 **Defaulter Profile Analysis**
    
    📈 **Correlation Analysis**
    
    🤖 **Multiple ML Models**
    
    🏆 **Automatic Best Model Selection**
    
    👤 **Customer Risk Prediction**
    
    📥 **Download Prediction Reports**
    """)

# DATASET UPLOAD

st.sidebar.markdown("---")

st.sidebar.markdown("### 📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset (Optional)",
    type=["csv", "xls", "xlsx"],
    help=(
        "Upload a CSV or Excel credit risk dataset. "
        "If no dataset is uploaded, LoanGuard will use "
        "a built-in demo dataset."))

st.sidebar.caption("Supported formats: CSV, XLS, XLSX")

# DATASET PATHS

BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDER = BASE_DIR / "data"
DEFAULT_DATASET = DATA_FOLDER / "default_credit_risk.csv"

# LOAD DATA

@st.cache_data
def load_data(uploaded_file=None):
    # USER UPLOADED DATASET
    if uploaded_file is not None:

        file_name = uploaded_file.name.lower()
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file format. "
                             "Please upload CSV or Excel file.")
        source = f"Uploaded Dataset: {uploaded_file.name}"

    # DEFAULT DATASET
    else:
        if os.path.exists(DEFAULT_DATASET):
            df = pd.read_csv(DEFAULT_DATASET)
            source = ("Default Dataset: " 
                      "default_credit_risk.csv")
        else:
            raise FileNotFoundError(f"Default dataset not found: {DEFAULT_DATASET}")
    return df, source

# LOAD DATASET

try:
    df, dataset_source = load_data(uploaded_file)
    if uploaded_file is not None:
        st.sidebar.success("✅ Custom dataset loaded")
        st.sidebar.info(f"📄 {uploaded_file.name}")
    else:
        st.sidebar.success("🗂️ Default dataset loaded")
        st.sidebar.info("📄 default_credit_risk.csv")
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.stop()

# DATASET INFORMATION
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Status")
st.sidebar.write(f"**Rows:** {df.shape[0]:,}")
st.sidebar.write(f"**Columns:** {df.shape[1]}")
st.sidebar.write(
    f"**Missing Values:** {df.isnull().sum().sum():,}"
)

# ABOUT APP

st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ About LoanGuard"):
    st.markdown(
        """
        **LoanGuard** is an Machine Learning based credit
        risk analytics platform.
        It analyzes customer financial profiles
        and predicts the probability of loan default.
        **Models Included:**
        - Logistic Regression
        - Decision Tree
        - Random Forest
        - Gradient Boosting
        - XGBoost
        - Support Vector Machine
        - K-Nearest Neighbors
        """)
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="
        text-align:center;
        color:gray;
        font-size:12px;
        padding:10px;
    ">
        <b>LoanGuard</b><br>
        Credit Risk Intelligence Platform<br><br>
        Powered by Machine Learning 🤖
    </div>
    """,
    unsafe_allow_html=True
)

# REQUIRED COLUMNS
required_columns = ["age", "monthly_income_bdt", "account_balance_bdt", "credit_score", "loan_amount_bdt", "loan_tenure_months",
                    "interest_rate_pct", "monthly_installment_bdt", "previous_loans",  "transaction_frequency_monthly", "gender",
                    "division", "district", "education", "employment_type", "account_type", "loan_type", "previous_default", "default"]
missing_columns = [col
    for col in required_columns
    if col not in df.columns]
if missing_columns:
    st.error("❌ The uploaded dataset is missing required columns:")
    st.write(missing_columns)
    st.info(
        "Please upload a dataset with the required columns "
        "or remove the uploaded file to use the default dataset.")
    st.stop()

# REMOVE ID COLUMNS FROM MODEL DATA

df = df.drop(columns=["customer_id", "customer_name"], errors="ignore")

# FEATURE ENGINEERING

df["loan_to_income_ratio"] = (df["loan_amount_bdt"] / (df["monthly_income_bdt"] * 12))
df["installment_to_income_ratio"] = (df["monthly_installment_bdt"] / df["monthly_income_bdt"])
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

# FEATURE LISTS

numeric_features = ["age", "monthly_income_bdt", "account_balance_bdt", "credit_score", "loan_amount_bdt", "loan_tenure_months", "interest_rate_pct",
                    "monthly_installment_bdt", "previous_loans", "transaction_frequency_monthly", "loan_to_income_ratio", "installment_to_income_ratio"]
categorical_features = ["gender", "division", "district", "education", "employment_type", "account_type", "loan_type", "previous_default"]

# PREPROCESSING

numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer(transformers=[("numeric", numeric_transformer, numeric_features), ("categorical", categorical_transformer, categorical_features)])

# FEATURES AND TARGET

X = df[numeric_features + categorical_features].copy()
y = df["default"].copy()

# CHECK TARGET

if y.nunique() < 2:
    st.error("The 'default' column must contain both " 
             "0 and 1 classes.")
    st.stop()
    
# TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# TRAIN MODELS

@st.cache_resource
def train_models(X_train, X_test, y_train, y_tes):
    models = {
        "Logistic Regression":
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Decision Tree":
            DecisionTreeClassifier(max_depth=6, min_samples_split=10, min_samples_leaf=5, class_weight="balanced", random_state=42),
        "Random Forest":
            RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=10, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting":
            GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42),
        "XGBoost":
            XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, min_child_weight=3, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1),
        "SVM":
            SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=42),
        "KNN":
            KNeighborsClassifier(n_neighbors=7, weights="distance")
    }

    trained_models = {}
    results = []
    roc_data = {}
    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        results.append({ "Model": name, "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob)})
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": roc_auc_score(y_test, y_prob)}

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    return (trained_models, results_df, roc_data)
    
# MODEL TRAINING
with st.spinner( "🤖 Training machine learning models..."): (trained_models, results_df, roc_data) = train_models(X_train, X_test, y_train, y_test)

# BEST MODEL
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
best_pred = best_model.predict(X_test)
best_prob = best_model.predict_proba(X_test)[:, 1]

# TABS

tabs = st.tabs([
    "📊 Dashboard",
    "🔍 EDA",
    "👥 Defaulter Analysis",
    "📈 Correlation",
    "🤖 Model Comparison",
    "🏆 Best Model",
    "👤 New Prediction"])


#  DASHBOARD
with tabs[0]:
    st.header("📊 Credit Risk Dashboard")
    st.info(f"Current Dataset: {dataset_source}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Total Defaulters", f"{total_defaulters:,}")
    col3.metric("Default Rate", f"{default_rate:.2f}%")
    col4.metric("Defaulted Loan", f"৳{total_defaulted_loan:,.0f}")
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Defaulter Income", f"৳{avg_defaulter_income:,.0f}")
    col2.metric("Avg Credit Score", f"{avg_defaulter_credit_score:.0f}")
    col3.metric("Avg Defaulter Age", f"{avg_defaulter_age:.1f}")
    col4.metric("Avg Defaulted Loan", f"৳{avg_defaulted_loan:,.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="default", ax=ax)
        ax.set_xticklabels(["Non-Defaulter", "Defaulter"])
        ax.set_title("Default Distribution")
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        counts = (df["default"].value_counts().sort_index())
        fig, ax = plt.subplots()
        ax.pie(counts, labels=["Non-Defaulter", "Defaulter"], autopct="%1.1f%%")
        ax.set_title("Default Percentage")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)


#  EDA
with tabs[1]:
    st.header("🔍 Exploratory Data Analysis")
    eda_variable = st.selectbox("Select Numerical Variable", numeric_features)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(data=df, x=eda_variable, hue="default", kde=True, bins=30, ax=ax)
        ax.set_title(f"Distribution of {eda_variable}")
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="default", y=eda_variable,  ax=ax)
        ax.set_xticklabels(["Non-Defaulter", "Defaulter"])
        ax.set_title(f"{eda_variable} by Default Status")
        st.pyplot(fig)
        plt.close(fig)
        
    st.markdown("---")

    categorical_variable = st.selectbox("Select Categorical Variable", categorical_features)

    category_counts = (df[categorical_variable].value_counts().head(15))
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=category_counts.values, y=category_counts.index, ax=ax)
    ax.set_title(f"Distribution of {categorical_variable}")
    st.pyplot(fig)
    plt.close(fig)

# DEFAULTER ANALYSIS
with tabs[2]:
    st.header("👥 Defaulter Profile Analysis")
    analysis_variable = st.selectbox("Select Variable", categorical_features)
    default_rate_analysis = (df.groupby(analysis_variable)["default"].mean().mul(100).sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=default_rate_analysis.values, y=default_rate_analysis.index, ax=ax)
    ax.set_xlabel("Default Rate (%)")
    ax.set_title(f"Default Rate by {analysis_variable}")
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(default_rate_analysis.reset_index().rename(columns={"default": "Default Rate (%)" }), use_container_width=True)

# CORRELATION
with tabs[3]:
    st.header("📈 Correlation Analysis")
    correlation = (df[numeric_features + ["default"]].corr())
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("Relationship with Default")

    default_corr = (correlation["default"].drop("default").sort_values(key=abs, ascending=False))
    correlation_table = (default_corr.reset_index())
    correlation_table.columns = ["Variable", "Correlation"]
    st.dataframe(correlation_table, use_container_width=True)


#  MODEL COMPARISON
with tabs[4]:
    st.header("🤖 Model Comparison")
    display_results = (results_df.copy().round(4))
    st.dataframe(display_results, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))

        sns.barplot(data=results_df, x="ROC-AUC", y="Model", ax=ax)
        ax.set_xlim(0, 1)
        ax.set_title("Model Performance - ROC AUC")
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=results_df, x="F1 Score", y="Model", ax=ax)
        ax.set_xlim(0, 1)
        ax.set_title("Model Performance - F1 Score")
        st.pyplot(fig)
        plt.close(fig)
        
    st.markdown("---")
    st.subheader("ROC Curves")
    fig, ax = plt.subplots(figsize=(10, 7))

    for name, data in roc_data.items():
        ax.plot(data["fpr"], data["tpr"],
            label=(f"{name} " 
                   f"(AUC = {data['auc']:.3f})"))

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.set_title("ROC Curve Comparison")
    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "⬇️ Download Model Results",
        results_df.to_csv(index=False),
        "model_comparison.csv",
        "text/csv",
        use_container_width=True)


# BEST MODEL
with tabs[5]:
    st.header("🏆 Best Model Performance")
    st.success(f"🏆 Selected Model: {best_model_name}")
    accuracy = accuracy_score(y_test, best_pred)
    precision = precision_score(y_test, best_pred, zero_division=0)
    recall = recall_score(y_test, best_pred, zero_division=0)
    f1 = f1_score(y_test, best_pred, zero_division=0)
    auc = roc_auc_score(y_test, best_prob)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{accuracy:.3f}")
    col2.metric("Precision", f"{precision:.3f}")
    col3.metric("Recall", f"{recall:.3f}")
    col4.metric("F1 Score", f"{f1:.3f}")
    col5.metric("ROC-AUC", f"{auc:.3f}")

    st.markdown("---")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, best_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Defaulter", "Defaulter"],
                yticklabels=["Non-Defaulter", "Defaulter"],
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Classification Report")

    report = classification_report(y_test, best_pred, target_names=["Non-Defaulter", "Defaulter"], 
                                   zero_division=0, output_dict=True)

    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    # Feature importance
    model_object = best_model.named_steps["model"]
    preprocessor_object = best_model.named_steps["preprocessor"]

    if hasattr(model_object, "feature_importances_"):
        feature_names = (preprocessor_object.get_feature_names_out())
        feature_importance = pd.DataFrame({"Feature": feature_names,
                                           "Importance": model_object.feature_importances_}).sort_values("Importance",ascending=False).head(20)

        st.markdown("---")
        st.subheader("🔎 Top Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=feature_importance, x="Importance", y="Feature", ax=ax)
        st.pyplot(fig)
        plt.close(fig)


# NEW CUSTOMER PREDICTION

with tabs[6]:
    st.header("👤 New Customer Risk Prediction")
    st.caption(
        f"Prediction is generated using the best model: "
        f"{best_model_name}")

    st.subheader("💰 Financial Information")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)

    with col2:
        monthly_income_bdt = st.number_input("Monthly Income (BDT)", min_value=1.0, value=50000.0, step=1000.0)

    with col3:
        account_balance_bdt = st.number_input("Account Balance (BDT)", min_value=0.0, value=100000.0, step=5000.0)

    col1, col2, col3 = st.columns(3)
    with col1:
        credit_score = st.number_input("Credit Score", min_value=0.0, max_value=1000.0, value=700.0)
    with col2:
        loan_amount_bdt = st.number_input("Loan Amount (BDT)", min_value=0.0, value=300000.0, step=10000.0)
    with col3:
        loan_tenure_months = st.number_input("Loan Tenure (Months)", min_value=1, max_value=360, value=36)

    col1, col2, col3 = st.columns(3)
    with col1:
        interest_rate_pct = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0)
    with col2:
        monthly_installment_bdt = st.number_input("Monthly Installment (BDT)", min_value=0.0, value=10000.0, step=500.0)
    with col3:
        previous_loans = st.number_input("Previous Loans", min_value=0, value=1)

    transaction_frequency_monthly = st.number_input("Monthly Transaction Frequency", min_value=0, value=20)

    st.markdown("---")
    st.subheader("👤 Customer Information")

    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", sorted(df["gender"].dropna().astype(str).unique()))
    with col2:
        division = st.selectbox("Division", sorted(df["division"].dropna().astype(str).unique()))
    with col3:
        district = st.selectbox("District", sorted(df["district"].dropna().astype(str).unique()))

    col1, col2, col3 = st.columns(3)
    with col1:
        education = st.selectbox("Education", sorted(df["education"].dropna().astype(str).unique()))
    with col2:
        employment_type = st.selectbox("Employment Type", sorted(df["employment_type"].dropna().astype(str).unique()))
    with col3:
        account_type = st.selectbox("Account Type",
            sorted(df["account_type"].dropna().astype(str).unique()))

    col1, col2 = st.columns(2)
    with col1:
        loan_type = st.selectbox("Loan Type",
            sorted(df["loan_type"].dropna().astype(str).unique()))
    with col2:
        previous_default = st.selectbox("Previous Default", sorted(df["previous_default"].dropna().astype(str).unique()))

    st.markdown("---")

    predict_button = st.button("🔮 Predict Credit Risk", type="primary", use_container_width=True)

    if predict_button:
        new_customer_df = pd.DataFrame({"age": [age], "monthly_income_bdt": [monthly_income_bdt], "account_balance_bdt": [account_balance_bdt],
                                        "credit_score": [credit_score], "loan_amount_bdt": [loan_amount_bdt], "loan_tenure_months": [loan_tenure_months],
                                        "interest_rate_pct": [interest_rate_pct], "monthly_installment_bdt": [monthly_installment_bdt], 
                                        "previous_loans": [previous_loans], "transaction_frequency_monthly": [transaction_frequency_monthly],
                                        "loan_to_income_ratio": [loan_amount_bdt / (monthly_income_bdt * 12)],
                                        "installment_to_income_ratio": [monthly_installment_bdt / monthly_income_bdt], "gender": [gender],
                                        "division": [division], "district": [district], "education": [education], "employment_type": [employment_type],
                                        "account_type": [account_type], "loan_type": [loan_type], "previous_default": [previous_default]})

        new_prediction = best_model.predict(new_customer_df)[0]
        new_probability = best_model.predict_proba(new_customer_df)[0, 1]

        if new_probability >= 0.70:
            risk_level = "HIGH RISK"
            risk_class = "risk-high"

        elif new_probability >= 0.40:
            risk_level = "MEDIUM RISK"
            risk_class = "risk-medium"

        else:
            risk_level = "LOW RISK"
            risk_class = "risk-low"

        prediction_label = ("Likely Defaulter" if new_prediction == 1 else "Likely Non-Defaulter")

        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", prediction_label)
        col2.metric("Default Probability", f"{new_probability:.2%}")
        col3.metric("Risk Level", risk_level)

        st.markdown(
            f"""
            <div class="{risk_class}">
                {risk_level}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.subheader("📊 Default Probability"))

        st.write(f"Estimated probability of default: "f"**{new_probability:.2%}**")

        report_df = pd.DataFrame({"Best_Model": [best_model_name], "Prediction": [prediction_label],
        "Default_Probability": [new_probability], "Risk_Level": [risk_level]})

        st.markdown("---")
        st.subheader("📋 Customer Risk Report")
        st.dataframe(report_df.style.format({"Default_Probability": "{:.2%}"}), use_container_width=True)
        st.download_button(
            "⬇️ Download Risk Report",
            report_df.to_csv(index=False),
            "customer_risk_prediction.csv",
            "text/csv",
            use_container_width=True)

# FOOTER
st.markdown("---")
st.markdown(
    """
    <div style="
        text-align:center;
        color:gray;
        padding:15px;
    ">
        <b>🏦 LoanGuard</b><br>
        Machine Learning based Credit Risk Prediction System<br>
        Data Science | Machine Learning 
    </div>
    """,
    unsafe_allow_html=True
)
