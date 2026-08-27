# ============================================================
# LOANGUARD - CREDIT RISK PREDICTION SYSTEM
# Streamlit Application
# ============================================================

import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

from xgboost import XGBClassifier


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LoanGuard - Credit Risk Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 40px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
    }

    .sub-title {
        font-size: 18px;
        text-align: center;
        color: #666666;
        margin-bottom: 30px;
    }

    .risk-high {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 25px;
        font-weight: bold;
        color: #990000;
    }

    .risk-medium {
        background-color: #fff0b3;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 25px;
        font-weight: bold;
        color: #996600;
    }

    .risk-low {
        background-color: #ccffcc;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 25px;
        font-weight: bold;
        color: #006600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🏦 LoanGuard - Credit Risk Prediction System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Machine Learning Based Credit Default Prediction & Risk Assessment'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏦 LoanGuard")

st.sidebar.markdown(
    """
    ### Navigation

    Explore the following sections:

    - 📊 Dashboard
    - 🔍 EDA
    - 👥 Defaulter Analysis
    - 📈 Correlation
    - 🤖 Model Comparison
    - 🏆 Best Model
    - 👤 New Customer Prediction
    """
)

st.sidebar.markdown("---")

st.sidebar.subheader("📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your own dataset",
    type=["csv", "xls", "xlsx"],
    help=(
        "Upload a credit risk dataset. "
        "If no file is uploaded, the built-in default "
        "dataset will be used."
    )
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "age",
    "monthly_income_bdt",
    "account_balance_bdt",
    "credit_score",
    "loan_amount_bdt",
    "loan_tenure_months",
    "interest_rate_pct",
    "monthly_installment_bdt",
    "previous_loans",
    "transaction_frequency_monthly",
    "gender",
    "division",
    "district",
    "education",
    "employment_type",
    "account_type",
    "loan_type",
    "previous_default",
    "loan_status",
    "default"
]


# ============================================================
# FIND DEFAULT DATASET
# ============================================================

def find_default_dataset():

    possible_paths = []

    try:
        app_directory = Path(__file__).resolve().parent

        possible_paths.append(
            app_directory
            / "data"
            / "default_credit_risk.csv"
        )

    except Exception:
        app_directory = Path.cwd()

    possible_paths.append(
        Path.cwd()
        / "data"
        / "default_credit_risk.csv"
    )

    possible_paths.append(
        app_directory
        / "default_credit_risk.csv"
    )

    possible_paths.append(
        Path.cwd()
        / "default_credit_risk.csv"
    )

    for path in possible_paths:

        if path.exists() and path.is_file():
            return path

    return None


# ============================================================
# LOAD UPLOADED DATA
# ============================================================

@st.cache_data
def load_uploaded_data(uploaded_file):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):

        try:
            return pd.read_csv(uploaded_file)

        except UnicodeDecodeError:

            uploaded_file.seek(0)

            return pd.read_csv(
                uploaded_file,
                encoding="latin1"
            )

    elif file_name.endswith(".xlsx"):

        return pd.read_excel(uploaded_file)

    elif file_name.endswith(".xls"):

        return pd.read_excel(uploaded_file)

    else:

        raise ValueError(
            "Unsupported file format."
        )


# ============================================================
# LOAD DEFAULT DATA
# ============================================================

@st.cache_data
def load_default_data(dataset_path):

    if dataset_path is None:
        return None

    try:

        return pd.read_csv(dataset_path)

    except UnicodeDecodeError:

        return pd.read_csv(
            dataset_path,
            encoding="latin1"
        )


# ============================================================
# LOAD DATASET
# ============================================================

if uploaded_file is not None:

    try:

        df = load_uploaded_data(
            uploaded_file
        )

        dataset_source = (
            f"Uploaded dataset: {uploaded_file.name}"
        )

        st.sidebar.success(
            f"Using uploaded dataset: {uploaded_file.name}"
        )

    except Exception as e:

        st.error(
            f"❌ Error loading uploaded dataset: {e}"
        )

        st.stop()

else:

    default_dataset_path = find_default_dataset()

    if default_dataset_path is None:

        st.sidebar.error(
            "Default dataset not found."
        )

        st.error(
            """
            ## ❌ Default dataset not found

            The application could not find:

            `data/default_credit_risk.csv`

            Your GitHub repository should have:

            ```
            LoanGuard/
            ├── app.py
            ├── requirements.txt
            └── data/
                └── default_credit_risk.csv
            ```
            """
        )

        st.info(
            "You can also upload your dataset from the sidebar."
        )

        with st.expander("🔎 Debug Information"):

            st.write(
                "**Current working directory:**"
            )

            st.code(str(Path.cwd()))

            try:

                st.write(
                    "**Application directory:**"
                )

                st.code(
                    str(
                        Path(__file__)
                        .resolve()
                        .parent
                    )
                )

                st.write(
                    "**Expected dataset path:**"
                )

                st.code(
                    str(
                        Path(__file__)
                        .resolve()
                        .parent
                        / "data"
                        / "default_credit_risk.csv"
                    )
                )

            except Exception as debug_error:

                st.write(
                    f"Debug information unavailable: "
                    f"{debug_error}"
                )

        st.stop()

    try:

        df = load_default_data(
            str(default_dataset_path)
        )

        dataset_source = (
            "Built-in default dataset"
        )

        st.sidebar.success(
            "Using built-in default dataset."
        )

    except Exception as e:

        st.error(
            f"❌ Error loading default dataset: {e}"
        )

        st.stop()


st.sidebar.caption(dataset_source)


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        "❌ The dataset is missing required columns."
    )

    st.write(
        "**Missing columns:**"
    )

    st.write(
        missing_columns
    )

    st.info(
        "Your dataset must contain all required "
        "credit-risk variables."
    )

    st.stop()


# ============================================================
# REMOVE IDENTIFIER COLUMNS
# ============================================================

df = df.drop(
    columns=[
        "customer_id",
        "customer_name"
    ],
    errors="ignore"
)


# ============================================================
# CLEAN TARGET
# ============================================================

df["default"] = pd.to_numeric(
    df["default"],
    errors="coerce"
)

df = df[
    df["default"].isin([0, 1])
].copy()

if len(df) == 0:

    st.error(
        "❌ No valid observations remain after cleaning."
    )

    st.stop()


if df["default"].nunique() < 2:

    st.error(
        "❌ The dataset must contain both "
        "default classes 0 and 1."
    )

    st.stop()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["loan_to_income_ratio"] = (
    df["loan_amount_bdt"]
    /
    (
        df["monthly_income_bdt"] * 12
    )
)

df["installment_to_income_ratio"] = (
    df["monthly_installment_bdt"]
    /
    df["monthly_income_bdt"].replace(
        0,
        np.nan
    )
)


df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


# ============================================================
# DEFAULTER INFORMATION
# ============================================================

defaulters = df[
    df["default"] == 1
].copy()

non_defaulters = df[
    df["default"] == 0
].copy()


total_customers = len(df)

total_defaulters = len(
    defaulters
)

total_non_defaulters = len(
    non_defaulters
)

default_rate = (
    total_defaulters
    /
    total_customers
) * 100


total_defaulted_loan = (
    defaulters[
        "loan_amount_bdt"
    ].sum()
)

avg_defaulted_loan = (
    defaulters[
        "loan_amount_bdt"
    ].mean()
)

avg_defaulter_income = (
    defaulters[
        "monthly_income_bdt"
    ].mean()
)

avg_defaulter_credit_score = (
    defaulters[
        "credit_score"
    ].mean()
)

avg_defaulter_age = (
    defaulters[
        "age"
    ].mean()
)


# ============================================================
# FEATURES
# ============================================================

numeric_features = [
    "age",
    "monthly_income_bdt",
    "account_balance_bdt",
    "credit_score",
    "loan_amount_bdt",
    "loan_tenure_months",
    "interest_rate_pct",
    "monthly_installment_bdt",
    "previous_loans",
    "transaction_frequency_monthly",
    "loan_to_income_ratio",
    "installment_to_income_ratio"
]


categorical_features = [
    "gender",
    "division",
    "district",
    "education",
    "employment_type",
    "account_type",
    "loan_type",
    "previous_default",
    "loan
