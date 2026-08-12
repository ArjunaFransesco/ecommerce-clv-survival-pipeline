import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

st.set_page_config(
    page_title="E-Commerce CLV & Customer Survival Engine",
    page_icon="📈",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1e3a8a; }
    .sub-text { font-size: 1.1rem; color: #475569; margin-bottom: 20px; }
    .metric-card { background-color: #f8fafc; border-radius: 10px; padding: 15px; border-left: 5px solid #3b82f6; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📈 E-Commerce Customer Lifetime Value & Survival Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Developed by <b>Arjuna Fransesco</b> | Machine Learning & Advanced Analytics Portfolio</div>', unsafe_allow_html=True)

# Load Metrics & Models
@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    clv_model = joblib.load(os.path.join(base_dir, "models", "clv_gradient_boosting.joblib"))
    churn_model = joblib.load(os.path.join(base_dir, "models", "churn_risk_classifier.joblib"))
    with open(os.path.join(base_dir, "reports", "metrics.json")) as f:
        metrics = json.load(f)
    return clv_model, churn_model, metrics

clv_model, churn_model, metrics = load_assets()

# Sidebar Inputs
st.sidebar.header("🎯 Customer Profile Simulator")
recency = st.sidebar.slider("Recency (Days since last purchase)", 1, 365, 30)
frequency = st.sidebar.slider("Frequency (Total orders count)", 1, 50, 6)
monetary = st.sidebar.number_input("Average Order Value ($)", min_value=10.0, max_value=2000.0, value=125.0, step=10.0)
tenure = st.sidebar.slider("Customer Tenure (Days)", 15, 1200, 240)
age = st.sidebar.slider("Customer Age", 18, 75, 32)
returns_rate = st.sidebar.slider("Returns Rate", 0.0, 0.8, 0.05, step=0.01)
channel = st.sidebar.selectbox("Acquisition Channel", ["Organic Search", "Paid Social", "Email Referral", "Google Ads", "Direct"])
membership = st.sidebar.selectbox("Membership Tier", ["Standard", "Bronze", "Silver", "Gold", "Platinum"])

# RFM Segment Estimation
r_score = 4 if recency <= 30 else (3 if recency <= 60 else (2 if recency <= 120 else 1))
f_score = 4 if frequency >= 10 else (3 if frequency >= 5 else (2 if frequency >= 2 else 1))
m_score = 4 if monetary >= 250 else (3 if monetary >= 120 else (2 if monetary >= 60 else 1))

if r_score >= 3 and f_score >= 3 and m_score >= 3:
    segment = "Champions / VIP"
elif r_score >= 3 and f_score >= 2:
    segment = "Loyal Customers"
elif r_score >= 3 and f_score == 1:
    segment = "Promising / New"
elif r_score <= 2 and f_score >= 3:
    segment = "At Risk High-Value"
elif r_score == 1 and f_score <= 2:
    segment = "Hibernating / Churned"
else:
    segment = "Potential Loyalist"

# Live Prediction
input_df = pd.DataFrame([{
    "Recency_Days": recency,
    "Frequency_Orders": frequency,
    "Monetary_Avg_Order_Value": monetary,
    "Tenure_Days": tenure,
    "Customer_Age": age,
    "Acquisition_Channel": channel,
    "Membership_Tier": membership,
    "Returns_Rate": returns_rate,
    "R_Score": r_score,
    "F_Score": f_score,
    "M_Score": m_score,
    "Customer_Segment": segment
}])

pred_clv = clv_model.predict(input_df)[0]
churn_prob = churn_model.predict_proba(input_df)[0][1]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Predicted 1-Year CLV", f"${pred_clv:,.2f}")
with col2:
    st.metric("Churn Risk Probability", f"{churn_prob * 100:.1f}%")
with col3:
    st.metric("Assigned Segment", segment)
with col4:
    st.metric("Model R² Score", f"{metrics['CLV_Regression']['R2_Score']:.3f}")

st.markdown("---")

tab1, tab2 = st.tabs(["📊 Cohort Analytics & Strategy", "⚙️ Model Specifications"])

with tab1:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("💡 Recommended Retention Action")
        if churn_prob > 0.65:
            st.error("🚨 High Churn Risk: Trigger immediate win-back campaign with VIP retention discount & dedicated concierge.")
        elif churn_prob > 0.35:
            st.warning("⚠️ Medium Churn Risk: Deploy personalized email sequence based on previous category preferences.")
        else:
            st.success("✅ Healthy Engagement: Target with cross-sell and loyalty tier upgrade incentives.")
        
        st.write(f"**Customer RFM Score:** `{r_score}{f_score}{m_score}`")
        st.write(f"**Estimated Return on Retention:** `${pred_clv * (1 - churn_prob):,.2f}`")

    with col_b:
        st.subheader("📈 Cohort Segment Overview")
        st.image("reports/clv_and_segment_distribution.png", use_container_width=True)

with tab2:
    st.json(metrics)
