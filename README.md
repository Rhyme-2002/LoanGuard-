# 🏦 LoanGuard - Credit Risk Prediction System

LoanGuard is a Machine Learning-based Credit Risk Prediction System designed to analyze customer financial information and predict the probability of loan default.

The project includes both an interactive **Streamlit web application** and a **general raw Python implementation** for credit risk analysis and machine learning model development.

## 🚀 Live Application

Try the deployed Streamlit application:

👉 **[LoanGuard Live App](https://bankloanguard.streamlit.app/)**

---

## 📌 Project Features

* 📊 Interactive Credit Risk Dashboard
* 🔍 Exploratory Data Analysis (EDA)
* 👥 Defaulter Profile Analysis
* 📈 Correlation Analysis
* 🤖 Multiple Machine Learning Models
* 🏆 Automatic Best Model Selection
* 📉 ROC Curve Comparison
* 🎯 Customer Credit Risk Prediction
* 📋 Customer Risk Report Generation
* 💾 Model Comparison Results Export
* 🖥️ Streamlit Web Application
* 🐍 General Raw Python Implementation

---

## 🤖 Machine Learning Models

The following classification models are implemented:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost
6. Support Vector Machine (SVM)
7. K-Nearest Neighbors (KNN)

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix
* Classification Report
* ROC Curve

The best model is automatically selected based on the highest **ROC-AUC score**.

---

## 📂 Project Structure

```text
LoanGuard/
│
├── .devcontainer/
│   └── devcontainer.json
│
├── data/
│   └── default_credit_risk.csv
│
├── outputs/
│   ├── customer_risk_prediction.csv
│   └── model_comparison.csv
│
├── app.py
├── loan_guard.py
├── requirements.txt
│
└── README.md
```

### File Description

| File / Folder                          | Description                                         |
| -------------------------------------- | --------------------------------------------------- |
| `app.py`                               | Streamlit interactive web application               |
| `loan_guard.py`                        | General raw Python implementation without Streamlit |
| `data/default_credit_risk.csv`         | Credit risk dataset                                 |
| `outputs/model_comparison.csv`         | Machine learning model performance results          |
| `outputs/customer_risk_prediction.csv` | Customer credit risk prediction report              |
| `requirements.txt`                     | Required Python libraries                           |
| `.devcontainer/`                       | Development container configuration                 |
|`data/default_credit_risk.csv`          | Default credit risk dataset used for analysis, model training, and prediction.                                                |

---

# 🖥️ Streamlit Application

The interactive application is built using Streamlit.

### Run locally

Clone the repository:

```bash
git clone https://github.com/Rhyme-2002/LoanGuard-.git
```

Move to the project directory:

```bash
cd LoanGuard
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🐍 General Python Version

The project also includes a general Python implementation without Streamlit.

The `loan_guard.py` file performs:

* Dataset loading
* Data cleaning
* Feature engineering
* Exploratory Data Analysis
* Defaulter analysis
* Correlation analysis
* Data preprocessing
* Train-test splitting
* Machine learning model training
* Model comparison
* Best model selection
* Customer risk prediction
* Report generation

Run the general Python version:

```bash
python loan_guard.py
```

---

# 📊 Feature Engineering

Two additional financial features are created:

### Loan-to-Income Ratio

```text
loan_to_income_ratio =
loan_amount_bdt / (monthly_income_bdt × 12)
```

### Installment-to-Income Ratio

```text
installment_to_income_ratio =
monthly_installment_bdt / monthly_income_bdt
```

These features help measure the customer's financial burden relative to income.

---

# 🎯 Credit Risk Prediction

For a new customer, the system predicts:

* Prediction Class

  * Likely Defaulter
  * Likely Non-Defaulter

* Default Probability

* Risk Level

Risk levels are classified as:

| Default Probability | Risk Level     |
| ------------------- | -------------- |
| ≥ 70%               | 🔴 High Risk   |
| 40% – 69.99%        | 🟡 Medium Risk |
| < 40%               | 🟢 Low Risk    |

---

# 📊 Dataset Features

The model uses customer demographic, financial, banking, and loan-related information.

### Numerical Features

* Age
* Monthly Income
* Account Balance
* Credit Score
* Loan Amount
* Loan Tenure
* Interest Rate
* Monthly Installment
* Previous Loans
* Transaction Frequency
* Loan-to-Income Ratio
* Installment-to-Income Ratio

### Categorical Features

* Gender
* Division
* District
* Education
* Employment Type
* Account Type
* Loan Type
* Previous Default

### Target Variable

```text
default
```

* `0` → Non-Defaulter
* `1` → Defaulter

---

# ⚙️ Data Preprocessing

The machine learning pipeline includes:

### Numerical Data

* Missing value imputation using the median
* Feature scaling using `StandardScaler`

### Categorical Data

* Missing value imputation using the most frequent value
* One-Hot Encoding
* Unknown category handling

The preprocessing steps are integrated using:

```text
ColumnTransformer
```

and

```text
Pipeline
```

---

# 📈 Model Evaluation

Each model is evaluated on the test dataset using:

```text
Accuracy
Precision
Recall
F1 Score
ROC-AUC
```

The results are saved in:

```text
outputs/model_comparison.csv
```

---

# 📋 Prediction Output

Customer prediction results are saved in:

```text
outputs/customer_risk_prediction.csv
```

The prediction report contains:

```text
Best_Model
Prediction
Default_Probability
Risk_Level
```

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Streamlit

---

# 📦 Installation

Install all required packages:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
openpyxl
```

---

# 🌐 Deployment

The Streamlit application is deployed online.

### 🔗 Live Demo

👉 **https://bankloanguard.streamlit.app/**

---

# 👨‍💻 Author

**Abu Sufiun Rhyme**

Data Analyst | Data Scientist | Applied Statistician

---

# ⭐ If You Like This Project

If you find this project useful, consider giving the repository a ⭐.
