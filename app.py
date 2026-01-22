import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load models
models = {
    "Linear SVM": joblib.load("models/svm_linear.pkl"),
    "Polynomial SVM": joblib.load("models/svm_poly.pkl"),
    "RBF SVM": joblib.load("models/svm_rbf.pkl")
}

# Title & description
st.title("💳 Smart Loan Approval System")
st.write("This system uses **Support Vector Machines (SVM)** to predict loan approval.")

# Sidebar inputs
st.sidebar.header("Applicant Details")

income = st.sidebar.number_input("Applicant Income", min_value=0, value=5000)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=150)
credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Yes", "No"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

credit_history = 1 if credit_history == "Yes" else 0

# Feature engineering
income_log = np.log1p(income)
loan_log = np.log1p(loan_amount)
total_income_log = np.log1p(income)

input_df = pd.DataFrame({
    "ApplicantIncome_log": [income_log],
    "LoanAmount_log": [loan_log],
    "TotalIncome_log": [total_income_log],
    "Loan_Amount_Term": [360],
    "Credit_History": [credit_history],
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

    if credit_history == 1:
        st.info("Applicant has good credit history and income stability.")
    else:
        st.warning("Poor credit history increases repayment risk.")
