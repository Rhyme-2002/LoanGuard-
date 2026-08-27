# ============================================================
# CREDIT RISK PREDICTION APP
# Streamlit Application
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

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
    page_title="Credit Risk Prediction",
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

    .metric-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
        text-align: center;
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
    '<div class="main-title">🏦 Credit Risk Prediction System</div>',
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

st.sidebar.title("🏦 Credit Risk System")

st.sidebar.markdown(
    """
    ### Navigation

    Use the tabs in the main panel to explore:

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

uploaded_file = st.sidebar.file_uploader(
    "Upload Credit Risk Dataset",
    type=["csv", "xls", "xlsx"]
)


# ============================================================
# LOAD DATA FUNCTION
# ============================================================

@st.cache_data
def load_data(uploaded_file):

    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):

        data = pd.read_csv(uploaded_file)

    elif file_name.endswith(".xls"):

        data = pd.read_excel(uploaded_file)

    elif file_name.endswith(".xlsx"):

        data = pd.read_excel(uploaded_file)

    else:

        raise ValueError(
            "Unsupported file format."
        )

    return data


# ============================================================
# LOAD DATA
# ============================================================

if uploaded_file is None:

    st.info(
        "👈 Please upload your credit risk dataset from the sidebar."
    )

    st.markdown(
        """
        ### Expected dataset

        Your dataset should contain columns similar to:

        `customer_id`, `customer_name`, `age`, `gender`,
        `division`, `district`, `education`,
        `employment_type`, `monthly_income_bdt`,
        `account_balance_bdt`, `credit_score`,
        `loan_amount_bdt`, `loan_tenure_months`,
        `interest_rate_pct`, `monthly_installment_bdt`,
        `previous_loans`, `transaction_frequency_monthly`,
        `account_type`, `loan_type`, `previous_default`,
        `loan_status`, `default`
        """
    )

    st.stop()


# ============================================================
# READ DATA
# ============================================================

try:

    df = load_data(uploaded_file)

except Exception as e:

    st.error(
        f"Error loading dataset: {e}"
    )

    st.stop()


# ============================================================
# BASIC VALIDATION
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


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    st.error(
        "The following required columns are missing:"
    )

    st.write(
        missing_columns
    )

    st.stop()


# ============================================================
# REMOVE CUSTOMER ID
# ============================================================

df = df.drop(
    columns=["customer_id"],
    errors="ignore"
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["loan_to_income_ratio"] = (

    df["loan_amount_bdt"]
    /
    (df["monthly_income_bdt"] * 12)

)


df["installment_to_income_ratio"] = (

    df["monthly_installment_bdt"]
    /
    df["monthly_income_bdt"]

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
# CREATE DEFAULTER DATASET
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

total_defaulters = len(defaulters)

total_non_defaulters = len(non_defaulters)

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
            OneHotEncoder(
                handle_unknown="ignore"
            )
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
# FEATURES AND TARGET
# ============================================================

X = df.drop(

    columns=[
        "customer_name",
        "default"
    ],

    errors="ignore"

)


y = df["default"]


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

            n_jobs=-1

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
def train_models(X_train, X_test, y_train, y_test):

    trained_models = {}

    results = {}

    roc_data = {}

    for name, model in models.items():

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


        trained_models[
            name
        ] = pipeline


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


    results_df = pd.DataFrame(
        list(results.values())
    )


    results_df = results_df.sort_values(

        by="ROC-AUC",

        ascending=False

    )


    return (

        trained_models,

        results_df,

        roc_data

    )


# ============================================================
# TRAINING SPINNER
# ============================================================

with st.spinner(
    "🤖 Training machine learning models..."
):

    (
        trained_models,
        results_df,
        roc_data
    ) = train_models(

        X_train,
        X_test,
        y_train,
        y_test

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
# TAB 1 — DASHBOARD
# ============================================================

with tabs[0]:

    st.header(
        "📊 Credit Risk Dashboard"
    )


    st.markdown(
        "### Portfolio Overview"
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

        st.pyplot(fig)


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

        st.pyplot(fig)


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
            int(df.isnull().sum().sum())
        )

        st.write(
            "**Duplicate Rows:**",
            int(df.duplicated().sum())
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

        st.pyplot(fig)


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

            f"{eda_variable}: "
            "Defaulters vs Non-Defaulters"

        )

        ax.set_xlabel(
            "Default"
        )

        st.pyplot(fig)


    st.markdown("---")


    st.subheader(
        "Categorical Variable Analysis"
    )


    categorical_variable = st.selectbox(

        "Select categorical variable",

        [

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

    )


    categorical_counts = (

        df[categorical_variable]
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


    st.pyplot(fig)


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
            analysis_variable
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

        y=default_rate_analysis.index,

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


    st.pyplot(fig)


    st.dataframe(

        default_rate_analysis
        .reset_index()
        .rename(
            columns={
                "default":
                    "Default Rate (%)"
            }
        ),

        use_container_width=True

    )


    st.markdown("---")


    st.subheader(
        "Defaulter Profile"
    )


    profile = pd.DataFrame({

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

            defaulters[
                "age"
            ].mean(),

            defaulters[
                "monthly_income_bdt"
            ].mean(),

            defaulters[
                "account_balance_bdt"
            ].mean(),

            defaulters[
                "credit_score"
            ].mean(),

            defaulters[
                "loan_amount_bdt"
            ].mean(),

            defaulters[
                "previous_loans"
            ].mean(),

            defaulters[
                "transaction_frequency_monthly"
            ].mean(),

            defaulters[
                "loan_to_income_ratio"
            ].mean(),

            defaulters[
                "installment_to_income_ratio"
            ].mean()

        ]

    })


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


    correlation = df[
        numeric_features + ["default"]
    ].corr()


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


    st.pyplot(fig)


    st.markdown("---")


    st.subheader(
        "Variables Most Associated with Default"
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
# TAB 5 — MODEL COMPARISON
# ============================================================

with tabs[4]:

    st.header(
        "🤖 Machine Learning Model Comparison"
    )


    st.markdown(
        "Models are ranked according to ROC-AUC."
    )


    display_results = results_df.copy()


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


    col1, col2 = st.columns(2)


    with col1:

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


        st.pyplot(fig)


    with col2:

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


        st.pyplot(fig)


    st.markdown("---")


    col1, col2 = st.columns(2)


    with col1:

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


        st.pyplot(fig)


    with col2:

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


        st.pyplot(fig)


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


    report_df = pd.DataFrame(
        report
    ).transpose()


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


    st.pyplot(fig)


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🔎 Feature Importance"
    )


    model_object = (

        best_model.named_steps[
            "model"
        ]

    )


    preprocessor_object = (

        best_model.named_steps[
            "preprocessor"
        ]

    )


    if hasattr(

        model_object,

        "feature_importances_"

    ):

        feature_names = (

            preprocessor_object
            .get_feature_names_out()

        )


        importance = (

            model_object
            .feature_importances_

        )


        feature_importance = pd.DataFrame({

            "Feature":
                feature_names,

            "Importance":
                importance

        })


        feature_importance = (

            feature_importance

            .sort_values(

                "Importance",

                ascending=False

            )

            .head(20)

        )


        st.dataframe(

            feature_importance,

            use_container_width=True

        )


        fig, ax = plt.subplots(

            figsize=(10, 8)

        )


        sns.barplot(

            data=feature_importance,

            x="Importance",

            y="Feature",

            ax=ax

        )


        ax.set_title(

            f"Top 20 Feature Importance - "
            f"{best_model_name}"

        )


        st.pyplot(fig)


    else:

        st.info(

            f"{best_model_name} does not provide "
            "built-in feature_importances_."

        )


    # ========================================================
    # SAMPLE TEST CUSTOMER
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📋 Sample Test Customer Predictions"
    )


    sample_customers = (

        X_test
        .head(10)
        .copy()

    )


    sample_actual = (

        y_test
        .loc[
            sample_customers.index
        ]

    )


    sample_prediction = (

        best_model
        .predict(
            sample_customers
        )

    )


    sample_probability = (

        best_model
        .predict_proba(
            sample_customers
        )[:, 1]

    )


    prediction_table = pd.DataFrame({

        "Actual":
            sample_actual.values,

        "Predicted":
            sample_prediction,

        "Default_Probability":
            sample_probability

    })


    prediction_table[
        "Risk_Level"
    ] = np.where(

        prediction_table[
            "Default_Probability"
        ] >= 0.70,

        "High Risk",

        np.where(

            prediction_table[
                "Default_Probability"
            ] >= 0.40,

            "Medium Risk",

            "Low Risk"

        )

    )


    st.dataframe(

        prediction_table.style.format({

            "Default_Probability":
                "{:.2%}"

        }),

        use_container_width=True

    )


# ============================================================
# TAB 7 — NEW CUSTOMER PREDICTION
# ============================================================

with tabs[6]:

    st.header(
        "👤 New Customer Credit Risk Prediction"
    )


    st.markdown(
        """
        Enter the customer's information below.
        The trained best-performing model will estimate
        the probability of default.
        """
    )


    # ========================================================
    # NUMERIC INPUTS
    # ========================================================

    st.subheader(
        "💰 Financial Information"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        age = st.number_input(

            "Age",

            min_value=18,

            max_value=100,

            value=30

        )


    with col2:

        monthly_income_bdt = st.number_input(

            "Monthly Income (BDT)",

            min_value=0.0,

            value=50000.0,

            step=1000.0

        )


    with col3:

        account_balance_bdt = st.number_input(

            "Account Balance (BDT)",

            min_value=0.0,

            value=100000.0,

            step=5000.0

        )


    col1, col2, col3 = st.columns(3)


    with col1:

        credit_score = st.number_input(

            "Credit Score",

            min_value=0.0,

            max_value=1000.0,

            value=700.0,

            step=1.0

        )


    with col2:

        loan_amount_bdt = st.number_input(

            "Loan Amount (BDT)",

            min_value=0.0,

            value=300000.0,

            step=10000.0

        )


    with col3:

        loan_tenure_months = st.number_input(

            "Loan Tenure (Months)",

            min_value=1,

            max_value=360,

            value=36

        )


    col1, col2, col3 = st.columns(3)


    with col1:

        interest_rate_pct = st.number_input(

            "Interest Rate (%)",

            min_value=0.0,

            max_value=100.0,

            value=10.0,

            step=0.1

        )


    with col2:

        monthly_installment_bdt = st.number_input(

            "Monthly Installment (BDT)",

            min_value=0.0,

            value=10000.0,

            step=500.0

        )


    with col3:

        previous_loans = st.number_input(

            "Previous Loans",

            min_value=0,

            max_value=100,

            value=1

        )


    transaction_frequency_monthly = st.number_input(

        "Monthly Transaction Frequency",

        min_value=0,

        max_value=1000,

        value=20

    )


    # ========================================================
    # CATEGORICAL INPUTS
    # ========================================================

    st.subheader(
        "👤 Customer Information"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        gender = st.selectbox(

            "Gender",

            sorted(
                df["gender"]
                .dropna()
                .astype(str)
                .unique()
            )

        )


    with col2:

        division = st.selectbox(

            "Division",

            sorted(
                df["division"]
                .dropna()
                .astype(str)
                .unique()
            )

        )


    with col3:

        district = st.selectbox(

            "District",

            sorted(
                df["district"]
                .dropna()
                .astype(str)
                .unique()
            )

        )


    col1, col2, col3 = st.columns(3)


    with col1:

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


    with col3:

        account_type = st.selectbox(

            "Account Type",

            sorted(
                df["account_type"]
                .dropna()
                .astype(str)
                .unique()
            )

        )


    col1, col2, col3 = st.columns(3)


    with col1:

        loan_type = st.selectbox(

            "Loan Type",

            sorted(
                df["loan_type"]
                .dropna()
                .astype(str)
                .unique()
            )

        )


    with col2:

        previous_default = st.selectbox(

            "Previous Default",

            sorted(
                df["previous_default"]
                .dropna()
                .astype(str)
                .unique()
            )

        )


    with col3:

        loan_status = st.selectbox(

            "Loan Status",

            sorted(
                df["loan_status"]
                .dropna()
                .astype(str)
                .unique()
            )

        )


    st.markdown("---")


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    predict_button = st.button(

        "🔮 Predict Credit Risk",

        type="primary",

        use_container_width=True

    )


    if predict_button:

        # ====================================================
        # CREATE CUSTOMER DATA
        # ====================================================

        new_customer = {

            "age":
                age,

            "monthly_income_bdt":
                monthly_income_bdt,

            "account_balance_bdt":
                account_balance_bdt,

            "credit_score":
                credit_score,

            "loan_amount_bdt":
                loan_amount_bdt,

            "loan_tenure_months":
                loan_tenure_months,

            "interest_rate_pct":
                interest_rate_pct,

            "monthly_installment_bdt":
                monthly_installment_bdt,

            "previous_loans":
                previous_loans,

            "transaction_frequency_monthly":
                transaction_frequency_monthly,

            "gender":
                gender,

            "division":
                division,

            "district":
                district,

            "education":
                education,

            "employment_type":
                employment_type,

            "account_type":
                account_type,

            "loan_type":
                loan_type,

            "previous_default":
                previous_default,

            "loan_status":
                loan_status

        }


        new_customer_df = pd.DataFrame(
            [new_customer]
        )


        # ====================================================
        # FEATURE ENGINEERING
        # ====================================================

        if monthly_income_bdt > 0:

            new_customer_df[
                "loan_to_income_ratio"
            ] = (

                loan_amount_bdt
                /
                (monthly_income_bdt * 12)

            )


            new_customer_df[
                "installment_to_income_ratio"
            ] = (

                monthly_installment_bdt
                /
                monthly_income_bdt

            )

        else:

            new_customer_df[
                "loan_to_income_ratio"
            ] = np.nan


            new_customer_df[
                "installment_to_income_ratio"
            ] = np.nan


        # ====================================================
        # HANDLE INF
        # ====================================================

        new_customer_df.replace(

            [np.inf, -np.inf],

            np.nan,

            inplace=True

        )


        # ====================================================
        # PREDICTION
        # ====================================================

        try:

            new_prediction = (

                best_model
                .predict(
                    new_customer_df
                )[0]

            )


            new_probability = (

                best_model
                .predict_proba(
                    new_customer_df
                )[0, 1]

            )


            # ================================================
            # RISK LEVEL
            # ================================================

            if new_probability >= 0.70:

                risk_level = "HIGH RISK"

            elif new_probability >= 0.40:

                risk_level = "MEDIUM RISK"

            else:

                risk_level = "LOW RISK"


            # ================================================
            # PREDICTION LABEL
            # ================================================

            if new_prediction == 1:

                prediction_label = (
                    "DEFAULTER"
                )

            else:

                prediction_label = (
                    "NON-DEFAULTER"
                )


            # ================================================
            # RESULT
            # ================================================

            st.markdown("---")

            st.subheader(
                "🎯 Credit Risk Result"
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(

                    "Best Model",

                    best_model_name

                )


            with col2:

                st.metric(

                    "Default Probability",

                    f"{new_probability:.2%}"

                )


            with col3:

                st.metric(

                    "Prediction",

                    prediction_label

                )


            # ================================================
            # RISK DISPLAY
            # ================================================

            if risk_level == "HIGH RISK":

                st.markdown(

                    '<div class="risk-high">'
                    '⚠️ HIGH RISK'
                    '</div>',

                    unsafe_allow_html=True

                )


            elif risk_level == "MEDIUM RISK":

                st.markdown(

                    '<div class="risk-medium">'
                    '⚠️ MEDIUM RISK'
                    '</div>',

                    unsafe_allow_html=True

                )


            else:

                st.markdown(

                    '<div class="risk-low">'
                    '✅ LOW RISK'
                    '</div>',

                    unsafe_allow_html=True

                )


            st.markdown("---")


            # ================================================
            # PROBABILITY BAR
            # ================================================

            st.subheader(
                "Default Probability"
            )


            st.progress(
                float(new_probability)
            )


            st.write(

                f"Probability of Default: "
                f"**{new_probability:.2%}**"

            )


            # ================================================
            # CUSTOMER RISK REPORT
            # ================================================

            new_customer_report = pd.DataFrame({

                "Model": [

                    best_model_name

                ],

                "Prediction": [

                    prediction_label

                ],

                "Default_Probability": [

                    new_probability

                ],

                "Risk_Level": [

                    risk_level

                ]

            })


            st.subheader(
                "📋 Customer Risk Report"
            )


            st.dataframe(

                new_customer_report.style.format({

                    "Default_Probability":
                        "{:.2%}"

                }),

                use_container_width=True

            )


            # ================================================
            # CUSTOMER INFORMATION
            # ================================================

            with st.expander(
                "👤 View Customer Information"
            ):

                st.dataframe(

                    new_customer_df,

                    use_container_width=True

                )


            # ================================================
            # DOWNLOAD REPORT
            # ================================================

            csv_data = (

                new_customer_report
                .to_csv(
                    index=False
                )

            )


            st.download_button(

                label="⬇️ Download Risk Report",

                data=csv_data,

                file_name="new_customer_prediction.csv",

                mime="text/csv",

                use_container_width=True

            )


        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(

    """
    <div style="text-align:center; color:gray;">

    <b>Credit Risk Prediction System</b><br>

    Machine Learning | Credit Risk Analytics | Data Science

    </div>
    """,

    unsafe_allow_html=True

)
