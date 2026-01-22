# =====================================
# TRAIN & SAVE SVM MODELS (PATH SAFE)
# =====================================

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# -------------------------------------
# Resolve BASE directory safely
# -------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------------------
# Load dataset
# -------------------------------------
df = pd.read_csv(DATA_PATH)

# Encode target
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Handle missing values
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])
df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

# Feature engineering
df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["ApplicantIncome_log"] = np.log1p(df["ApplicantIncome"])
df["LoanAmount_log"] = np.log1p(df["LoanAmount"])
df["TotalIncome_log"] = np.log1p(df["TotalIncome"])

numeric_features = [
    "ApplicantIncome_log",
    "LoanAmount_log",
    "TotalIncome_log",
    "Loan_Amount_Term",
    "Credit_History"
]

categorical_features = [
    "Self_Employed",
    "Property_Area"
]

X = df[numeric_features + categorical_features]
y = df["Loan_Status"]

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
    ]
)

# Models
models = {
    "linear": SVC(kernel="linear", probability=True),
    "poly": SVC(kernel="poly", degree=3, probability=True),
    "rbf": SVC(kernel="rbf", probability=True)
}

# Train & save
for name, svm in models.items():
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", svm)
    ])
    pipeline.fit(X, y)

    model_path = os.path.join(MODELS_DIR, f"svm_{name}.pkl")
    joblib.dump(pipeline, model_path)

print("✅ Models trained and saved successfully.")
