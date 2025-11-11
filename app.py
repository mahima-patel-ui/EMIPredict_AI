import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
df_input=None
# ------------------------------
# 📥 Download dataset from Google Drive if missing
# ------------------------------
import gdown, os

DATA_PATH = "sample_data/emi_prediction_dataset.csv"
DATA_URL =  "https://drive.google.com/uc/id=1YoxwH1XWbe4Oo3ixPb_tYZBYGc-Vm1jc" # replace with your file ID

os.makedirs("sample_data", exist_ok=True)

if not os.path.exists(DATA_PATH):
    st.info("⬇️ Downloading dataset from Google Drive...")
    gdown.download(DATA_URL, DATA_PATH, quiet=False)

# ------------------------------
# 🔽 Auto-Download Models from Google Drive (if missing)
# ------------------------------
import gdown

drive_models = {
    "XGBoost_classification.joblib": "https://drive.google.com/uc?id=1iQ18cb-saM34sCKGCLqP6jwbPCwplcgQ",
    "XGBoost_regression.joblib": "https://drive.google.com/uc?id=1xMKo4-stFBLnshCv22oTf5p4Et94fDxR",
    "feature_names.json": "https://drive.google.com/uc?id=1TZgnsgLQ16OpO5u84suFvQZ5Nqm1F_gk",  # optional if uploaded
    "scaler.joblib": "https://drive.google.com/uc?id=1zBqFCocWbvwZX4RNkuhZYaswxYhI9e0b"
}

os.makedirs("artifacts/models", exist_ok=True)

for fname, url in drive_models.items():
    dest = os.path.join("artifacts/models", fname)
    if not os.path.exists(dest):
        st.info(f"⬇️ Downloading {fname} from Google Drive...")
        gdown.download(url, dest, quiet=False)



# ------------------------------
# 🌟 App Configuration
# ------------------------------
st.set_page_config(page_title="EMIPredict AI", layout="wide")
st.title("💰 EMIPredict AI — Smart EMI Prediction App")

st.markdown("""
This interactive web app allows you to:
1. 📊 Explore the dataset (Quick EDA)
2. 🤖 Predict EMI eligibility
3. 💵 Estimate maximum EMI amount

> Built using Machine Learning models trained in **EMIPredict_AI.ipynb**.
""")

# ------------------------------
# 📁 Load Saved Artifacts
# ------------------------------
MODEL_DIR = "artifacts/models"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

# Attempt to load models and scaler
models = {}
scaler = None

if os.path.exists(MODEL_DIR):
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".joblib"):
            try:
                models[f] = joblib.load(os.path.join(MODEL_DIR, f))
            except Exception as e:
                st.warning(f"⚠️ Could not load model {f}: {e}")

if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
# ------------------------------
# 📂 Load Feature Names (Fix)
# ------------------------------
import json

FEATURE_PATH = os.path.join(MODEL_DIR, "feature_names.json")

feature_names = []
if os.path.exists(FEATURE_PATH):
    with open(FEATURE_PATH, "r") as f:
        feature_names = json.load(f)
    st.success(f"✅ Loaded {len(feature_names)} training feature names.")
else:
    st.warning("⚠️ feature_names.json not found in artifacts/models — predictions may mismatch.")


# ------------------------------
# 📂 Sidebar Navigation
# ------------------------------
st.sidebar.header("🔧 Navigation")
mode = st.sidebar.radio(
    "Choose a mode:",
    ["Overview", "Upload & Quick EDA", "Single Prediction"],
    index=0
)

# ------------------------------
# 🔹 Mode 1: Overview
# ------------------------------
if mode == "Overview":
    st.header("📘 Overview")
    st.write("""
    **EMIPredict AI** helps financial institutions predict EMI eligibility and estimate safe EMI amounts.
    Models and visualizations are pre-trained and stored inside the `artifacts/` folder.
    """)

    # Show available artifacts
    if os.path.exists(MODEL_DIR):
        st.subheader("🧠 Available Models")
        st.write(os.listdir(MODEL_DIR))

    if os.path.exists("artifacts/eda_charts"):
        st.subheader("📊 EDA Charts (Sample)")
        chart_files = os.listdir("artifacts/eda_charts")[:9]
        cols = st.columns(3)
        for i, chart in enumerate(chart_files):
            with cols[i % 3]:
                st.image(os.path.join("artifacts/eda_charts", chart), use_column_width=True)

    st.markdown("""
    ---
    💡 *Tip:* To view detailed model performance and insights, open the notebook `EMIPredict_AI.ipynb`.
    """)

# ------------------------------
# 🔹 Mode 2: Upload & Quick EDA
# ------------------------------
elif mode == "Upload & Quick EDA":
    st.header("📊 Quick Exploratory Data Analysis (EDA)")

    uploaded = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
        st.dataframe(df.head())

        st.subheader("📈 Basic Statistics")
        st.write(df.describe())

        # Numeric histograms
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 0:
            st.subheader("🔹 Distribution of Numeric Features")
            selected = st.multiselect("Select numeric columns for visualization", num_cols, num_cols[:3])
            for col in selected:
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, bins=30, ax=ax, color="skyblue")
                ax.set_title(f"Distribution of {col}")
                st.pyplot(fig)
        else:
            st.info("No numeric columns found for plotting.")

# ------------------------------
# 🔹 Mode 3: Single Prediction
# ------------------------------
elif mode == "Single Prediction":
    st.header("🤖 EMI Prediction")

    st.markdown("""
    Upload a **single applicant record** in CSV format (with the same features used in training),
    **or** manually enter feature values below 👇
    """)

    uploaded = st.file_uploader("📤 Upload single-row CSV file", type=["csv"])

    # --- Manual Input Form ---
    with st.expander("📋 Or fill values manually"):
        salary = st.number_input("Monthly Salary (₹)", min_value=0.0, value=50000.0, step=1000.0)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=720)
        current_emi = st.number_input("Current EMI Amount (₹)", min_value=0.0, value=5000.0, step=100.0)
        other_expenses = st.number_input("Other Monthly Expenses (₹)", min_value=0.0, value=10000.0, step=100.0)
        years_emp = st.number_input("Years of Employment", min_value=0.0, value=3.0, step=0.5)
        dependents = st.number_input("Number of Dependents", min_value=0, value=1, step=1)

    # --- Prepare Input Data ---
    if uploaded is not None:
        df_input = pd.read_csv(uploaded)
        st.success("✅ File uploaded successfully!")
    else:
        df_input = pd.DataFrame([{
            "monthly_salary": salary,
            "credit_score": credit_score,
            "current_emi_amount": current_emi,
            "other_monthly_expenses": other_expenses,
            "years_of_employment": years_emp,
            "dependents": dependents
        }])

    # --- Display Input Preview ---
    st.subheader("📄 Input Data Preview")
    st.dataframe(df_input)

    # --- Prediction Button ---
    if st.button("🔮 Predict EMI Details"):
        try:
            # Align features with training features
            if 'feature_names' in locals() and len(feature_names) > 0:
                for col in feature_names:
                    if col not in df_input.columns:
                        df_input[col] = 0
                df_input = df_input[feature_names]
            else:
                st.warning("⚠️ Using available columns only (feature_names.json not loaded).")

            # Select numeric columns only
            X = df_input.select_dtypes(include=[np.number])

            # Apply scaling if available
            if scaler is not None:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X.values

            # Get models
            clf = next((m for n, m in models.items() if "classification" in n.lower()), None)
            reg = next((m for n, m in models.items() if "regression" in n.lower()), None)

            # --- Classification Prediction ---
            if clf is not None:
                pred_class = clf.predict(X_scaled)[0]
                st.success(f"✅ EMI Eligibility: **{pred_class}**")
                try:
                    prob = clf.predict_proba(X_scaled)[0]
                    st.progress(int(prob.max() * 100))
                    st.write(f"Confidence: {prob.max():.2%}")
                except:
                    pass
            else:
                st.warning("⚠️ Classification model not found.")

            # --- Regression Prediction ---
            if reg is not None:
                pred_emi = reg.predict(X_scaled)[0]
                st.success(f"💵 Predicted Maximum EMI: ₹{pred_emi:,.2f}")
            else:
                st.warning("⚠️ Regression model not found.")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
