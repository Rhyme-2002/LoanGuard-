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
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🏦 LoanGuard - Credit Risk Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Machine Learning Based Credit Default Prediction & Risk Assessment</div>',
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
    help="Upload a credit risk dataset. If no file is uploaded, the built-in default dataset will be used."
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

        return pd.read_csv(
            dataset_path
        )

    except UnicodeDecodeError:

        return pd.read_csv(
            dataset_path,
            encoding="latin1"
        )


# ============================================================
# DATA LOADING
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

    default_dataset_path = (
        find_default_dataset()
    )

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

            LoanGuard/
            ├── app.py
            ├── requirements.txt
            └── data/
                └── default_credit_risk.csv
            """
        )

        st.info(
            "You can also upload your dataset from the sidebar."
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


st.sidebar.caption(
    dataset_source
)


# ============================================================
# DATA CLEANING
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.lower()
)


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

    st.stop()


df = df.drop(
    columns=[
        "customer_id",
        "customer_name"
    ],
    errors="ignore"
)


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


df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


# ============================================================
# BASIC STATISTICS
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
    "loan_status"
]


X = df.drop(
    columns=["default"]
)


y = df["default"].astype(
    int
)


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
# PREPROCESSING
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
    ]
)


# ============================================================
# MODELS
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
        list(results.values())
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
# RUN TRAINING
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
# MODEL ERRORS
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


best_pred = best_model.predict(
    X_test
)


best_prob = best_model.predict_proba(
    X_test
)[:, 1]


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
# TAB 1 - DASHBOARD
# ============================================================

with tabs[0]:

    st.header(
        "📊 Credit Risk Dashboard"
    )

    st.info(
        f"Dataset: {dataset_source}"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Total Defaulters",
        f"{total_defaulters:,}"
    )

    col3.metric(
        "Default Rate",
        f"{default_rate:.2f}%"
    )

    col4.metric(
        "Defaulted Loan",
        f"৳{total_defaulted_loan:,.0f}"
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


    st.subheader(
        "Dataset Preview"
    )


    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# ============================================================
# TAB 2 - EDA
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
            f"{eda_variable}: Default Comparison"
        )


        st.pyplot(
            fig,
            clear_figure=True
        )


# ============================================================
# TAB 3 - DEFAULTER ANALYSIS
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


# ============================================================
# TAB 4 - CORRELATION
# ============================================================

with tabs[3]:

    st.header(
        "📈 Correlation Analysis"
    )


    correlation = (
        df[
            numeric_features + ["default"]
        ]
        .corr(
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


    default_corr = (
        correlation["default"]
        .drop("default")
        .sort_values(
            key=abs,
            ascending=False
        )
    )


    correlation_table = (
        default_corr
        .reset_index()
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
# TAB 5 - MODEL COMPARISON
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


    # ROC-AUC BAR PLOT

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


    # RECALL

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


    # F1 SCORE

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


    # ROC CURVES

    st.subheader(
        "ROC Curves"
    )


    fig, ax = plt.subplots(
        figsize=(10, 7)
    )


    for name, data in roc_data.items():

        ax.plot(
            data["fpr"],
            data["tpr"],
            label=f'{name} (AUC = {data["auc"]:.3f})'
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
        "ROC Curve Comparison"
    )


    ax.legend()


    st.pyplot(
        fig,
        clear_figure=True
    )


# ============================================================
# TAB 6 - BEST MODEL
# ============================================================

with tabs[5]:

    st.header(
        "🏆 Best Model Performance"
    )


    st.success(
        f"🏆 Best Model: {best_model_name}"
    )


    col1, col2, col3, col4, col5 = st.columns(5)


    col1.metric(
        "Accuracy",
        f"{accuracy:.3f}"
    )


    col2.metric(
        "Precision",
        f"{precision:.3f}"
    )


    col3.metric(
        "Recall",
        f"{recall:.3f}"
    )


    col4.metric(
        "F1 Score",
        f"{f1:.3f}"
    )


    col5.metric(
        "ROC-AUC",
        f"{roc_auc:.3f}"
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
        figsize=(7, 5)
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Non-Default",
            "Default"
        ],
        yticklabels=[
            "Non-Default",
            "Default"
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


    st.subheader(
        "Classification Report"
    )


    report = classification_report(
        y_test,
        best_pred,
        output_dict=True,
        zero_division=0
    )


    report_df = pd.DataFrame(
        report
    ).transpose()


    st.dataframe(
        report_df,
        use_container_width=True
    )


# ============================================================
# TAB 7 - NEW CUSTOMER PREDICTION
# ============================================================

with tabs[6]:

    st.header(
        "👤 New Customer Credit Risk Prediction"
    )


    st.info(
        f"Predictions are generated using the best model: {best_model_name}"
    )


    st.subheader(
        "Customer Information"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )


        monthly_income_bdt = st.number_input(
            "Monthly Income (BDT)",
            min_value=1.0,
            value=50000.0
        )


        account_balance_bdt = st.number_input(
            "Account Balance (BDT)",
            min_value=0.0,
            value=100000.0
        )


        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650
        )


    with col2:

        loan_amount_bdt = st.number_input(
            "Loan Amount (BDT)",
            min_value=1.0,
            value=300000.0
        )


        loan_tenure_months = st.number_input(
            "Loan Tenure (Months)",
            min_value=1,
            max_value=360,
            value=36
        )


        interest_rate_pct = st.number_input(
            "Interest Rate (%)",
            min_value=0.0,
            value=12.0
        )


        monthly_installment_bdt = st.number_input(
            "Monthly Installment (BDT)",
            min_value=0.0,
            value=10000.0
        )


    with col3:

        previous_loans = st.number_input(
            "Previous Loans",
            min_value=0,
            value=1
        )


        transaction_frequency_monthly = st.number_input(
            "Monthly Transaction Frequency",
            min_value=0,
            value=20
        )


        gender = st.selectbox(
            "Gender",
            sorted(
                df["gender"]
                .dropna()
                .astype(str)
                .unique()
            )
        )


        division = st.selectbox(
            "Division",
            sorted(
                df["division"]
                .dropna()
                .astype(str)
                .unique()
            )
        )


    col1, col2, col3 = st.columns(3)


    with col1:

        district = st.selectbox(
            "District",
            sorted(
                df["district"]
                .dropna()
                .astype(str)
                .unique()
            )
        )


        education = st.selectbox(
            "Education",
            sorted(
                df["education"]
                .dropna()
                .astype(str)
                .unique()
            )
        )


    with col2:

        employment_type = st.selectbox(
            "Employment Type",
            sorted(
                df["employment_type"]
                .dropna()
                .astype(str)
                .unique()
            )
        )


        account_type = st.selectbox(
            "Account Type",
            sorted(
                df["account_type"]
                .dropna()
                .astype(str)
                .unique()
            )
        )


    with col3:

        loan_type = st.selectbox(
            "Loan Type",
            sorted(
                df["loan_type"]
                .dropna()
                .astype(str)
                .unique()
            )
        )


        previous_default = st.selectbox(
            "Previous Default",
            sorted(
                df["previous_default"]
                .dropna()
                .astype(str)
                .unique()
            )
        )


        loan_status = st.selectbox(
            "Loan Status",
            sorted(
                df["loan_status"]
                .dropna()
                .astype(str)
                .unique()
            )
        )


    if st.button(
        "🔍 Predict Credit Risk",
        type="primary",
        use_container_width=True
    ):


        loan_to_income_ratio = (
            loan_amount_bdt
            /
            (
                monthly_income_bdt
                * 12
            )
        )


        installment_to_income_ratio = (
            monthly_installment_bdt
            /
            monthly_income_bdt
        )


        new_customer = pd.DataFrame(
            [
                {
                    "age": age,
                    "monthly_income_bdt": monthly_income_bdt,
                    "account_balance_bdt": account_balance_bdt,
                    "credit_score": credit_score,
                    "loan_amount_bdt": loan_amount_bdt,
                    "loan_tenure_months": loan_tenure_months,
                    "interest_rate_pct": interest_rate_pct,
                    "monthly_installment_bdt": monthly_installment_bdt,
                    "previous_loans": previous_loans,
                    "transaction_frequency_monthly": transaction_frequency_monthly,
                    "gender": gender,
                    "division": division,
                    "district": district,
                    "education": education,
                    "employment_type": employment_type,
                    "account_type": account_type,
                    "loan_type": loan_type,
                    "previous_default": previous_default,
                    "loan_status": loan_status,
                    "loan_to_income_ratio": loan_to_income_ratio,
                    "installment_to_income_ratio": installment_to_income_ratio
                }
            ]
        )


        prediction = best_model.predict(
            new_customer
        )[0]


        probability = best_model.predict_proba(
            new_customer
        )[0, 1]


        st.markdown("---")


        st.subheader(
            "Prediction Result"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Probability of Default",
                f"{probability:.2%}"
            )


        with col2:

            st.metric(
                "Predicted Class",
                "Default"
                if prediction == 1
                else "Non-Default"
            )


        if probability >= 0.70:

            st.markdown(
                f"""
                <div class="risk-high">
                    🔴 HIGH CREDIT RISK<br>
                    Default Probability: {probability:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )


        elif probability >= 0.40:

            st.markdown(
                f"""
                <div class="risk-medium">
                    🟡 MEDIUM CREDIT RISK<br>
                    Default Probability: {probability:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )


        else:

            st.markdown(
                f"""
                <div class="risk-low">
                    🟢 LOW CREDIT RISK<br>
                    Default Probability: {probability:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )


        st.markdown("---")


        st.subheader(
            "Customer Input Summary"
        )


        st.dataframe(
            new_customer,
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🏦 LoanGuard | Machine Learning Based Credit Risk Prediction System"
)
