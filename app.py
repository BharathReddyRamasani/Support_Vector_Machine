# =====================================
# SMART LOAN APPROVAL SYSTEM
# =====================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------
# Resolve BASE directory safely
# -------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load models
models = {
    "Linear SVM": joblib.load(os.path.join(MODELS_DIR, "svm_linear.pkl")),
    "Polynomial SVM": joblib.load(os.path.join(MODELS_DIR, "svm_poly.pkl")),
    "RBF SVM": joblib.load(os.path.join(MODELS_DIR, "svm_rbf.pkl"))
}

# -------------------------------------
# App UI
# -------------------------------------
st.title("💳 Smart Loan Approval System")
st.write("This system uses **Support Vector Machines (SVM)** to predict loan approval.")

st.sidebar.header("Applicant Details")

income = st.sidebar.number_input("Applicant Income", min_value=0, value=5000)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=150)
credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Yes", "No"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

credit_history_val = 1 if credit_history == "Yes" else 0

# Feature engineering
input_df = pd.DataFrame({
    "ApplicantIncome_log": [np.log1p(income)],
    "LoanAmount_log": [np.log1p(loan_amount)],
    "TotalIncome_log": [np.log1p(income)],
    "Loan_Amount_Term": [360],
    "Credit_History": [credit_history_val],
    "Self_Employed": [employment],
    "Property_Area": [property_area]
})

# Model selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio(
    "Choose SVM Kernel",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

# Prediction
if st.button("🔍 Check Loan Eligibility"):
    model = models[model_choice]
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"**Kernel Used:** {model_choice}")
    st.write(f"**Confidence Score:** {confidence:.2f}")

    if credit_history_val == 1:
        st.info("Applicant shows good credit history and income stability.")
    else:
        st.warning("Poor credit history increases repayment risk.")
