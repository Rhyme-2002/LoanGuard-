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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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


st.set_page_config(
    page_title="LoanGuard - Credit Risk Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


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
# TITLE
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
        "If no file is uploaded, the built-in default dataset "
        "will be used."
    )
)


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


def find_default_dataset():
    """
    Search for the default dataset.

    Expected structure:

        LoanGuard/
        ├── app.py
        ├── requirements.txt
        └── data/
            └── default_credit_risk.csv
    """

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
# LOAD DATA
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

            Your GitHub repository should have exactly this structure:

            ```
            LoanGuard/
            │
            ├── app.py
            ├── requirements.txt
            │
            └── data/
                └── default_credit_risk.csv
            ```

            Make sure the file is committed and pushed to GitHub.
            """
        )

        st.info(
            "You can also upload your dataset from the sidebar."
        )

        with st.expander("🔎 Debug Information"):

            try:

                st.write(
                    "**Current working directory:**"
                )

                st.code(
                    str(Path.cwd())
                )

                st.write(
                    "**Application directory:**"
                )

                st.code(
                    str(
                        Path(__file__).resolve().parent
                    )
                )

                st.write(
                    "**Expected dataset path:**"
                )

                st.code(
                    str(
                        Path(__file__).resolve().parent
                        / "data"
                        / "default_credit_risk.csv"
                    )
                )

                st.write(
                    "**Files in application directory:**"
                )

                for item in (
                    Path(__file__).resolve().parent
                ).iterdir():

                    st.write(
                        str(item)
                    )

            except Exception as debug_error:

                st.write(
                    f"Debug information unavailable: {debug_error}"
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


# ============================================================
# DATASET SOURCE
# ============================================================

st.sidebar.caption(
    dataset_source
)


# ============================================================
# BASIC DATA CLEANING
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
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
        """
        Your dataset must contain all required
        credit-risk variables.
        """
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
# CONVERT TARGET TO NUMERIC
# ============================================================

df["default"] = pd.to_numeric(
    df["default"],
    errors="coerce"
)


# ============================================================
# REMOVE INVALID TARGET ROWS
# ============================================================

df = df[
    df["default"].isin([0, 1])
].copy()


# ============================================================
# CHECK TARGET
# ============================================================

if len(df) == 0:

    st.error(
        "❌ No valid observations remain after cleaning."
    )

    st.stop()


if df["default"].nunique() < 2:

    st.error(
        "❌ The dataset must contain both default classes 0 and 1."
    )

    st.stop()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["loan_to_income_ratio"] = (
    df["loan_amount_bdt"]
    /
    (
        df["monthly_income_bdt"]
        * 12
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


# ============================================================
# HANDLE INFINITE VALUES
# ============================================================

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


# ============================================================
# DEFAULTER DATA
# ============================================================

defaulters = df[
    df["default"] == 1
].copy()


non_defaulters = df[
    df["default"] == 0
].copy()


# ============================================================
# BASIC STATISTICS
# ============================================================

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
# FEATURE LIST
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
    "loan_status"
]


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[
        "default"
    ],
    errors="ignore"
)


X = X.drop(
    columns=[
        "customer_name",
        "customer_id"
    ],
    errors="ignore"
)


y = df[
    "default"
].astype(int)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# ONE HOT ENCODER
# ============================================================

try:

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

except TypeError:

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )


# ============================================================
# PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            encoder
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ],
    remainder="drop"
)


# ============================================================
# ML MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=10,
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist"
        ),

    "SVM":
        SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=7,
            weights="distance"
        )
}


# ============================================================
# TRAIN MODELS
# ============================================================

@st.cache_resource
def train_models(
    X_train,
    X_test,
    y_train,
    y_test
):

    trained_models = {}
    results = {}
    roc_data = {}
    errors = {}

    for name, model in models.items():

        try:

            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor
                    ),
                    (
                        "model",
                        model
                    )
                ]
            )

            pipeline.fit(
                X_train,
                y_train
            )

            trained_models[name] = pipeline

            y_pred = pipeline.predict(
                X_test
            )

            y_prob = pipeline.predict_proba(
                X_test
            )[:, 1]

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            precision = precision_score(
                y_test,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                zero_division=0
            )

            roc_auc = roc_auc_score(
                y_test,
                y_prob
            )

            results[name] = {
                "Model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "ROC-AUC": roc_auc
            }

            fpr, tpr, thresholds = roc_curve(
                y_test,
                y_prob
            )

            roc_data[name] = {
                "fpr": fpr,
                "tpr": tpr,
                "auc": roc_auc
            }

        except Exception as e:

            errors[name] = str(e)

    if not results:

        raise RuntimeError(
            "All machine learning models failed to train."
        )

    results_df = pd.DataFrame(
        list(
            results.values()
        )
    )

    results_df = results_df.sort_values(
        by="ROC-AUC",
        ascending=False
    ).reset_index(
        drop=True
    )

    return (
        trained_models,
        results_df,
        roc_data,
        errors
    )


# ============================================================
# TRAINING
# ============================================================

with st.spinner(
    "🤖 Training machine learning models..."
):

    try:

        (
            trained_models,
            results_df,
            roc_data,
            model_errors
        ) = train_models(
            X_train,
            X_test,
            y_train,
            y_test
        )

    except Exception as e:

        st.error(
            f"❌ Model training failed: {e}"
        )

        st.stop()


# ============================================================
# SHOW MODEL ERRORS
# ============================================================

if model_errors:

    with st.expander(
        "⚠️ Models with training errors"
    ):

        for model_name, error in model_errors.items():

            st.write(
                f"**{model_name}:** {error}"
            )


# ============================================================
# BEST MODEL
# ============================================================

best_model_name = (
    results_df.iloc[0]["Model"]
)


best_model = (
    trained_models[
        best_model_name
    ]
)


# ============================================================
# BEST MODEL PREDICTIONS
# ============================================================

best_pred = (
    best_model.predict(
        X_test
    )
)


best_prob = (
    best_model.predict_proba(
        X_test
    )[:, 1]
)


accuracy = accuracy_score(
    y_test,
    best_pred
)


precision = precision_score(
    y_test,
    best_pred,
    zero_division=0
)


recall = recall_score(
    y_test,
    best_pred,
    zero_division=0
)


f1 = f1_score(
    y_test,
    best_pred,
    zero_division=0
)


roc_auc = roc_auc_score(
    y_test,
    best_prob
)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "📊 Dashboard",
        "🔍 EDA",
        "👥 Defaulter Analysis",
        "📈 Correlation",
        "🤖 Model Comparison",
        "🏆 Best Model",
        "👤 New Customer Prediction"
    ]
)


# ============================================================
# TAB 1 — DASHBOARD
# ============================================================

with tabs[0]:

    st.header(
        "📊 Credit Risk Dashboard"
    )

    st.info(
        f"Dataset: {dataset_source}"
    )

    st.subheader(
        "Portfolio Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

    with col2:

        st.metric(
            "Total Defaulters",
            f"{total_defaulters:,}"
        )

    with col3:

        st.metric(
            "Default Rate",
            f"{default_rate:.2f}%"
        )

    with col4:

        st.metric(
            "Defaulted Loan",
            f"৳{total_defaulted_loan:,.0f}"
        )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Avg Defaulter Income",
            f"৳{avg_defaulter_income:,.0f}"
        )

    with col2:

        st.metric(
            "Avg Credit Score",
            f"{avg_defaulter_credit_score:.1f}"
        )

    with col3:

        st.metric(
            "Avg Defaulter Age",
            f"{avg_defaulter_age:.1f}"
        )

    with col4:

        st.metric(
            "Avg Defaulted Loan",
            f"৳{avg_defaulted_loan:,.0f}"
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        sns.countplot(
            data=df,
            x="default",
            ax=ax
        )

        ax.set_title(
            "Default Distribution"
        )

        ax.set_xlabel(
            "Default (0 = No, 1 = Yes)"
        )

        ax.set_ylabel(
            "Number of Customers"
        )

        st.pyplot(
            fig,
            clear_figure=True
        )

    with col2:

        default_counts = (
            df["default"]
            .value_counts()
            .sort_index()
        )

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.pie(
            default_counts.values,
            labels=[
                "Non-Defaulter",
                "Defaulter"
            ],
            autopct="%1.1f%%"
        )

        ax.set_title(
            "Default Percentage"
        )

        st.pyplot(
            fig,
            clear_figure=True
        )

    st.markdown("---")

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader(
        "Dataset Information"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "**Rows:**",
            df.shape[0]
        )

        st.write(
            "**Columns:**",
            df.shape[1]
        )

    with col2:

        st.write(
            "**Missing Values:**",
            int(
                df.isnull()
                .sum()
                .sum()
            )
        )

        st.write(
            "**Duplicate Rows:**",
            int(
                df.duplicated()
                .sum()
            )
        )


# ============================================================
# TAB 2 — EDA
# ============================================================

with tabs[1]:

    st.header(
        "🔍 Exploratory Data Analysis"
    )

    eda_variable = st.selectbox(
        "Select numerical variable",
        numeric_features
    )

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        sns.histplot(
            data=df,
            x=eda_variable,
            bins=25,
            kde=True,
            ax=ax
        )

        ax.set_title(
            f"Distribution of {eda_variable}"
        )

        st.pyplot(
            fig,
            clear_figure=True
        )

    with col2:

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        sns.boxplot(
            data=df,
            x="default",
            y=eda_variable,
            ax=ax
        )

        ax.set_title(
            f"{eda_variable}: Defaulters vs Non-Defaulters"
        )

        ax.set_xlabel(
            "Default"
        )

        st.pyplot(
            fig,
            clear_figure=True
        )

    st.markdown("---")

    st.subheader(
        "Categorical Variable Analysis"
    )

    categorical_variable = st.selectbox(
        "Select categorical variable",
        categorical_features
    )

    categorical_counts = (
        df[
            categorical_variable
        ]
        .astype(str)
        .value_counts()
        .head(15)
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.barplot(
        x=categorical_counts.values,
        y=categorical_counts.index,
        ax=ax
    )

    ax.set_title(
        f"Distribution of {categorical_variable}"
    )

    ax.set_xlabel(
        "Number of Customers"
    )

    ax.set_ylabel(
        categorical_variable
    )

    st.pyplot(
        fig,
        clear_figure=True
    )


# ============================================================
# TAB 3 — DEFAULTER ANALYSIS
# ============================================================

with tabs[2]:

    st.header(
        "👥 Defaulter Analysis"
    )

    analysis_variable = st.selectbox(
        "Select variable for default analysis",
        [
            "gender",
            "division",
            "education",
            "employment_type",
            "loan_type",
            "previous_default"
        ]
    )

    default_rate_analysis = (
        df.groupby(
            analysis_variable,
            dropna=False
        )["default"]
        .mean()
        .mul(100)
        .sort_values(
            ascending=False
        )
    )

    st.subheader(
        f"Default Rate by {analysis_variable}"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.barplot(
        x=default_rate_analysis.values,
        y=default_rate_analysis.index.astype(str),
        ax=ax
    )

    ax.set_xlabel(
        "Default Rate (%)"
    )

    ax.set_ylabel(
        analysis_variable
    )

    ax.set_title(
        f"Default Rate by {analysis_variable}"
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    st.dataframe(
        default_rate_analysis
        .reset_index()
        .rename(
            columns={
                "default": "Default Rate (%)"
            }
        ),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader(
        "Defaulter Profile"
    )

    profile = pd.DataFrame(
        {
            "Metric": [
                "Average Age",
                "Average Income",
                "Average Account Balance",
                "Average Credit Score",
                "Average Loan Amount",
                "Average Previous Loans",
                "Average Transaction Frequency",
                "Average Loan-to-Income Ratio",
                "Average Installment-to-Income Ratio"
            ],
            "Value": [
                defaulters["age"].mean(),
                defaulters["monthly_income_bdt"].mean(),
                defaulters["account_balance_bdt"].mean(),
                defaulters["credit_score"].mean(),
                defaulters["loan_amount_bdt"].mean(),
                defaulters["previous_loans"].mean(),
                defaulters["transaction_frequency_monthly"].mean(),
                defaulters["loan_to_income_ratio"].mean(),
                defaulters["installment_to_income_ratio"].mean()
            ]
        }
    )

    st.dataframe(
        profile,
        use_container_width=True
    )


# ============================================================
# TAB 4 — CORRELATION
# ============================================================

with tabs[3]:

    st.header(
        "📈 Correlation Analysis"
    )

    correlation = (
        df[
            numeric_features + ["default"]
        ].corr(
            numeric_only=True
        )
    )

    fig, ax = plt.subplots(
        figsize=(14, 10)
    )

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax
    )

    ax.set_title(
        "Correlation Matrix"
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    st.markdown("---")

    st.subheader(
        "Variables Most Associated with Default"
    )

    default_corr = (
        correlation[
            "default"
        ]
        .drop("default")
        .sort_values(
            key=abs,
            ascending=False
        )
    )

    correlation_table = (
        default_corr.reset_index()
    )

    correlation_table.columns = [
        "Variable",
        "Correlation"
    ]

    st.dataframe(
        correlation_table,
        use_container_width=True
    )


# ============================================================
# TAB 5 — MODEL COMPARISON
# ============================================================

with tabs[4]:

    st.header(
        "🤖 Machine Learning Model Comparison"
    )

    st.markdown(
        "Models are ranked according to ROC-AUC."
    )

    display_results = (
        results_df.copy()
    )

    for col in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]:

        display_results[col] = (
            display_results[col]
            .round(4)
        )

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    sns.barplot(
        data=results_df,
        x="ROC-AUC",
        y="Model",
        ax=ax
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_title(
        "Model Comparison - ROC-AUC"
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    sns.barplot(
        data=results_df,
        x="Recall",
        y="Model",
        ax=ax
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_title(
        "Defaulter Recall Comparison"
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    sns.barplot(
        data=results_df,
        x="F1 Score",
        y="Model",
        ax=ax
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_title(
        "F1 Score Comparison"
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for name, data in roc_data.items():

        ax.plot(
            data["fpr"],
            data["tpr"],
            label=(
                f"{name} "
                f"(AUC={data['auc']:.3f})"
            )
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC Curves"
    )

    ax.legend(
        fontsize=8
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    st.download_button(
        label="⬇️ Download Model Comparison",
        data=results_df.to_csv(
            index=False
        ),
        file_name="model_comparison_results.csv",
        mime="text/csv"
    )


# ============================================================
# TAB 6 — BEST MODEL
# ============================================================

with tabs[5]:

    st.header(
        "🏆 Best Credit Risk Model"
    )

    st.success(
        f"Best Model: {best_model_name}"
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy:.3f}"
        )

    with col2:

        st.metric(
            "Precision",
            f"{precision:.3f}"
        )

    with col3:

        st.metric(
            "Recall",
            f"{recall:.3f}"
        )

    with col4:

        st.metric(
            "F1 Score",
            f"{f1:.3f}"
        )

    with col5:

        st.metric(
            "ROC-AUC",
            f"{roc_auc:.3f}"
        )

    st.markdown("---")

    st.subheader(
        "Classification Report"
    )

    report = classification_report(
        y_test,
        best_pred,
        target_names=[
            "Non-Defaulter",
            "Defaulter"
        ],
        zero_division=0,
        output_dict=True
    )

    report_df = (
        pd.DataFrame(
            report
        ).transpose()
    )

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader(
        "Confusion Matrix"
    )

    cm = confusion_matrix(
        y_test,
        best_pred
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Non-Defaulter",
            "Defaulter"
        ],
        yticklabels=[
            "Non-Defaulter",
            "Defaulter"
        ],
        ax=ax
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_title(
        f"Confusion Matrix - {best_model_name}"
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    st.markdown("---")

    st.subheader(
        "🔎 Feature Importance"
    )

    model_object = (
        best_model.named_steps["model"]
    )

    preprocessor_object = (
        best_model.named_steps["preprocessor"]
    )

    if hasattr(
        model_object,
        "feature_importances_"
    ):

        feature_names = (
            preprocessor_object
            .
