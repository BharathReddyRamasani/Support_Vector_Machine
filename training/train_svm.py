import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv("../data/train.csv")

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

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
    ]
)

models = {
    "linear": SVC(kernel="linear", probability=True),
    "poly": SVC(kernel="poly", degree=3, probability=True),
    "rbf": SVC(kernel="rbf", probability=True)
}

for name, svm in models.items():
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", svm)
    ])
    pipeline.fit(X, y)
    joblib.dump(pipeline, f"../models/svm_{name}.pkl")

print("Models trained and saved successfully.")
