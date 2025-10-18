# ======================================================
# 🌟 PHASE 7 — Real-Time Inference + Streamlit Interface
# ======================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import xgboost as xgb

st.set_page_config(page_title="Risk-Aware Hybrid AI", layout="wide")

st.title("🧠 Risk-Aware Hybrid AI — Real-Time Risk Prediction")
st.markdown("""
This interface allows you to input new system log data or model features  
and get **real-time risk level predictions** using the hybrid deep ensemble + XGBoost model.
""")

# === Load trained model and supporting data ===
@st.cache_resource
def load_model():
    model = joblib.load("xgboost_ensemble_model.pkl")
    ensemble_features = pd.read_csv("ensemble_features.csv")
    return model, ensemble_features

try:
    xgb_model, sample_features = load_model()
    st.success("✅ Model and features loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model or data: {e}")
    st.stop()

# === Sidebar for feature input ===
st.sidebar.header("🔧 Input Features")
st.sidebar.write("Provide feature values below or use random example")

if st.sidebar.button("Use Random Example"):
    input_data = sample_features.sample(1).iloc[0].to_dict()
else:
    input_data = {col: st.sidebar.number_input(f"{col}", float(sample_features[col].mean()))
                  for col in sample_features.columns}

input_df = pd.DataFrame([input_data])

# === Predict risk level ===
if st.button("⚡ Run Prediction"):
    st.subheader("📊 Model Prediction Results")

    # Make prediction
    pred_prob = xgb_model.predict_proba(input_df)[0]
    pred_label = np.argmax(pred_prob)

    st.write(f"**Predicted Risk Level:** 🟢 {pred_label}")
    st.write(f"**Confidence Scores:** {np.round(pred_prob, 3)}")

    # === SHAP Explainability ===
    st.subheader("🧩 SHAP Explainability")
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(input_df)

    # SHAP force plot
    shap_fig, ax = plt.subplots()
    shap.waterfall_plot(shap.Explanation(values=shap_values.values[0],
                                         base_values=shap_values.base_values[0],
                                         data=input_df.iloc[0],
                                         feature_names=input_df.columns))
    st.pyplot(shap_fig)

    # Feature importance summary
    st.subheader("📈 Feature Importance Summary")
    shap_summary, ax2 = plt.subplots()
    shap.summary_plot(shap_values, input_df, show=False)
    st.pyplot(shap_summary)

    st.success("✅ Prediction and explainability complete!")

# === Footer ===
st.markdown("---")
st.caption("Developed as part of *Risk-Aware Hybrid AI* Project — Phase 7 Deployment")
