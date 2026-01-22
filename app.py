# # =====================================
# # SMART LOAN APPROVAL SYSTEM
# # =====================================

# import os
# import joblib
# import numpy as np
# import pandas as pd
# import streamlit as st

# # -------------------------------------
# # Resolve BASE directory safely
# # -------------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODELS_DIR = os.path.join(BASE_DIR, "models")

# # Load models
# models = {
#     "Linear SVM": joblib.load(os.path.join(MODELS_DIR, "svm_linear.pkl")),
#     "Polynomial SVM": joblib.load(os.path.join(MODELS_DIR, "svm_poly.pkl")),
#     "RBF SVM": joblib.load(os.path.join(MODELS_DIR, "svm_rbf.pkl"))
# }

# # -------------------------------------
# # App UI
# # -------------------------------------
# st.title("💳 Smart Loan Approval System")
# st.write("This system uses **Support Vector Machines (SVM)** to predict loan approval.")

# st.sidebar.header("Applicant Details")

# income = st.sidebar.number_input("Applicant Income", min_value=0, value=5000)
# loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=150)
# credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])
# employment = st.sidebar.selectbox("Employment Status", ["Yes", "No"])
# property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# credit_history_val = 1 if credit_history == "Yes" else 0

# # Feature engineering
# input_df = pd.DataFrame({
#     "ApplicantIncome_log": [np.log1p(income)],
#     "LoanAmount_log": [np.log1p(loan_amount)],
#     "TotalIncome_log": [np.log1p(income)],
#     "Loan_Amount_Term": [360],
#     "Credit_History": [credit_history_val],
#     "Self_Employed": [employment],
#     "Property_Area": [property_area]
# })

# # Model selection
# st.sidebar.header("Model Selection")
# model_choice = st.sidebar.radio(
#     "Choose SVM Kernel",
#     ["Linear SVM", "Polynomial SVM", "RBF SVM"]
# )

# # Prediction
# if st.button("🔍 Check Loan Eligibility"):
#     model = models[model_choice]
#     prediction = model.predict(input_df)[0]
#     confidence = model.predict_proba(input_df)[0][prediction]

#     if prediction == 1:
#         st.success("✅ Loan Approved")
#     else:
#         st.error("❌ Loan Rejected")

#     st.write(f"**Kernel Used:** {model_choice}")
#     st.write(f"**Confidence Score:** {confidence:.2f}")

#     if credit_history_val == 1:
#         st.info("Applicant shows good credit history and income stability.")
#     else:
#         st.warning("Poor credit history increases repayment risk.")





import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI Loan Underwriting Platform",
    page_icon="🏦",
    layout="wide"
)

# =====================================================
# ENTERPRISE DARK THEME (CSS)
# =====================================================
st.markdown("""
<style>
body {
    background-color: #0b0f14;
}
.block-container {
    padding: 2rem 3rem;
}
.section {
    background: #121826;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
}
.header-title {
    font-size: 42px;
    font-weight: 800;
}
.header-sub {
    color: #94a3b8;
    font-size: 18px;
}
.kpi {
    background: #0f172a;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}
.kpi h2 {
    margin: 0;
    font-size: 28px;
}
.kpi span {
    color: #94a3b8;
}
.approved {
    background: linear-gradient(135deg,#064e3b,#10b981);
    border-radius: 18px;
    padding: 32px;
    color: white;
    font-size: 30px;
    font-weight: 800;
    text-align: center;
}
.rejected {
    background: linear-gradient(135deg,#7f1d1d,#ef4444);
    border-radius: 18px;
    padding: 32px;
    color: white;
    font-size: 30px;
    font-weight: 800;
    text-align: center;
}
.action-btn button {
    background: linear-gradient(90deg,#6366f1,#ec4899);
    color: white;
    font-size: 20px;
    padding: 14px;
    border-radius: 14px;
}
.explain {
    background: #0f172a;
    border-left: 6px solid #6366f1;
    border-radius: 14px;
    padding: 20px;
}
.badge {
    background: #1e293b;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODELS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

models = {
    "Linear SVM": joblib.load(os.path.join(MODELS_DIR, "svm_linear.pkl")),
    "Polynomial SVM": joblib.load(os.path.join(MODELS_DIR, "svm_poly.pkl")),
    "RBF SVM": joblib.load(os.path.join(MODELS_DIR, "svm_rbf.pkl")),
}

# =====================================================
# SIDEBAR – CONFIGURATION
# =====================================================
st.sidebar.markdown("## ⚙️ Underwriting Config")

kernel = st.sidebar.radio(
    "SVM Kernel",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

st.sidebar.markdown(
    "<div class='badge'>💡 RBF handles complex borrower patterns</div>",
    unsafe_allow_html=True
)

# =====================================================
# HEADER
# =====================================================
st.markdown("<div class='header-sub'>AI-Powered Credit Risk Assessment</div>", unsafe_allow_html=True)
st.markdown("<div class='header-title'>Smart Loan Approval System</div>", unsafe_allow_html=True)

# =====================================================
# INPUT – FINANCIALS
# =====================================================
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("### 💰 Financial Profile")

c1, c2 = st.columns(2)
with c1:
    income = st.number_input("Monthly Income ($)", min_value=0, value=5000)
with c2:
    loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=140)

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# INPUT – BACKGROUND
# =====================================================
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("### 🧾 Applicant Background")

c3, c4, c5 = st.columns(3)
with c3:
    credit = st.selectbox("Credit History", ["Clean / Repaid", "Past Defaults"])
with c4:
    employment = st.selectbox("Employment Type", ["Salaried", "Self Employed"])
with c5:
    property_area = st.selectbox("Property Location", ["Urban", "Semiurban", "Rural"])

credit_val = 1 if credit == "Clean / Repaid" else 0
employment_val = "Yes" if employment == "Self Employed" else "No"

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# ACTION
# =====================================================
st.markdown("<div class='action-btn'>", unsafe_allow_html=True)
run = st.button("🚀 Run AI Risk Assessment", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PREDICTION
# =====================================================
if run:
    model = models[kernel]

    input_df = pd.DataFrame({
        "ApplicantIncome_log": [np.log1p(income)],
        "LoanAmount_log": [np.log1p(loan_amount)],
        "TotalIncome_log": [np.log1p(income)],
        "Loan_Amount_Term": [360],
        "Credit_History": [credit_val],
        "Self_Employed": [employment_val],
        "Property_Area": [property_area]
    })

    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0][prediction]

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("### 📊 Underwriting Decision")

    d1, d2 = st.columns([2, 1])

    with d1:
        if prediction == 1:
            st.markdown("<div class='approved'>✅ APPROVED<br><span style='font-size:16px'>Low Credit Risk</span></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='rejected'>❌ REJECTED<br><span style='font-size:16px'>High Credit Risk</span></div>", unsafe_allow_html=True)

    with d2:
        st.markdown("<div class='kpi'>", unsafe_allow_html=True)
        st.markdown("### 🤖 AI Confidence")
        st.progress(float(confidence))
        st.markdown(f"<h2>{confidence*100:.1f}%</h2>", unsafe_allow_html=True)
        st.markdown(f"<span>Model Certainty</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # =================================================
    # EXPLANATION
    # =================================================
    st.markdown("<div class='section explain'>", unsafe_allow_html=True)
    st.markdown("### ⚖️ Risk Explanation")

    if credit_val == 1:
        st.write(
            "The AI underwriting engine classified this applicant as **Low Risk**. "
            "Income stability and repayment history indicate a strong ability to meet loan obligations."
        )
    else:
        st.write(
            "The applicant shows **Elevated Credit Risk** due to adverse credit history, "
            "increasing probability of delinquency or default."
        )

    st.markdown("</div>", unsafe_allow_html=True)
