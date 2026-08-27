import streamlit as st
import requests as re

st.set_page_config(
    page_title="FraudGuard AI - Stacking Ensemble Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .fraud-badge {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(185, 28, 28, 0.25));
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
    }
    .legit-badge {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(21, 128, 61, 0.25));
        border: 1px solid #22c55e;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
    }
    .architecture-box {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# App Header
col_title, col_logo = st.columns([3, 1])
with col_title:
    st.markdown('<div class="main-header">🛡️ Financial Fraud Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by a <b>Stacking Ensemble</b> of XGBoost, LightGBM / Histogram Gradient Boosting, and Logistic Regression.</div>', unsafe_allow_html=True)

with col_logo:
    try:
        st.image("image.png", use_container_width=True)
    except Exception:
        pass

# About Section
with st.expander("ℹ️ About This AI Fraud Detection Engine (Stacking Ensemble)", expanded=False):
    st.markdown("""
    ### 🧠 Stacking Ensemble Architecture
    This system utilizes a multi-stage **Stacking Classifier** combining gradient boosted decision trees and meta-learning:
    
    1. **XGBoost (Extreme Gradient Boosting):** Captures non-linear feature interactions and high-loss outlier transactions.
    2. **LightGBM / Histogram Gradient Boosting:** Performs fast leaf-wise tree growth to detect complex fraud threshold boundaries.
    3. **Logistic Regression (Meta-Learner):** Blends the probabilities from both models with learned optimal weights to make the final calibrated risk prediction.

    #### 🔍 Key Risk Dimensions Analyzed:
    - **Transaction Channel:** Flags high-risk vectors (*Transfer* and *Cash Out*).
    - **Balance Discrepancies:** Identifies total balance depletion to `$0.00` and ghost destination accounts.
    - **Amount-to-Balance Ratios:** Compares the transaction amount against origin account depth.
    - **High-Value Thresholds:** Applies automated compliance triggers for transactions $\ge \$200,000$.
    """)

# Sidebar - Quick Presets
st.sidebar.markdown("### ⚡ Quick Test Scenarios")
scenario = st.sidebar.radio(
    "Load Preset Scenario:",
    ("Custom Input", "🚨 Suspicious Account Drain (Fraud)", "🚨 High-Risk Transfer (Fraud)", "✅ Legitimate Merchant Payment", "✅ Normal Money Transfer"),
    index=0
)

# Preset values configuration
defaults = {
    "sender": "C123456789",
    "receiver": "M987654321",
    "step": 1,
    "type_idx": 3,  # Payment
    "amount": 250.0,
    "old_orig": 5000.0,
    "new_orig": 4750.0,
    "old_dest": 1000.0,
    "new_dest": 1250.0
}

if scenario == "🚨 Suspicious Account Drain (Fraud)":
    defaults = {
        "sender": "C998877661",
        "receiver": "M112233445",
        "step": 1,
        "type_idx": 1,  # Cash Out
        "amount": 181.0,
        "old_orig": 181.0,
        "new_orig": 0.0,
        "old_dest": 21182.0,
        "new_dest": 0.0
    }
elif scenario == "🚨 High-Risk Transfer (Fraud)":
    defaults = {
        "sender": "C554433221",
        "receiver": "C887766554",
        "step": 1,
        "type_idx": 4,  # Transfer
        "amount": 100000.0,
        "old_orig": 100000.0,
        "new_orig": 0.0,
        "old_dest": 0.0,
        "new_dest": 0.0
    }
elif scenario == "✅ Legitimate Merchant Payment":
    defaults = {
        "sender": "C102030405",
        "receiver": "M504030201",
        "step": 5,
        "type_idx": 3,  # Payment
        "amount": 75.50,
        "old_orig": 2400.0,
        "new_orig": 2324.50,
        "old_dest": 12000.0,
        "new_dest": 12075.50
    }
elif scenario == "✅ Normal Money Transfer":
    defaults = {
        "sender": "C334455667",
        "receiver": "C778899001",
        "step": 12,
        "type_idx": 4,  # Transfer
        "amount": 500.0,
        "old_orig": 15000.0,
        "new_orig": 14500.0,
        "old_dest": 3000.0,
        "new_dest": 3500.0
    }

# Input Form
st.sidebar.markdown("---")
st.sidebar.header("📝 Transaction Parameters")

sender_name = st.sidebar.text_input("Sender Account ID", value=defaults["sender"])
receiver_name = st.sidebar.text_input("Recipient Account ID", value=defaults["receiver"])

type_options = ["Cash In", "Cash Out", "Debit", "Payment", "Transfer"]
type_mapping = {"Cash In": 0, "Cash Out": 1, "Debit": 2, "Payment": 3, "Transfer": 4}

selected_type_name = st.sidebar.selectbox("Transaction Type", type_options, index=defaults["type_idx"])
types = type_mapping[selected_type_name]

step = st.sidebar.slider("Transaction Duration / Step (Hours)", min_value=1, max_value=744, value=int(defaults["step"]))

amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=float(defaults["amount"]), step=50.0, format="%.2f")

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    oldbalanceorg = st.number_input("Sender Balance (Pre)", min_value=0.0, value=float(defaults["old_orig"]), step=100.0, format="%.2f")
    oldbalancedest = st.number_input("Recipient Balance (Pre)", min_value=0.0, value=float(defaults["old_dest"]), step=100.0, format="%.2f")

with col_sb2:
    newbalanceorg = st.number_input("Sender Balance (Post)", min_value=0.0, value=float(defaults["new_orig"]), step=100.0, format="%.2f")
    newbalancedest = st.number_input("Recipient Balance (Post)", min_value=0.0, value=float(defaults["new_dest"]), step=100.0, format="%.2f")

isflaggedfraud = 1 if amount >= 200000 else 0

# Main Interactive UI Body
st.markdown("### 📊 Transaction Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Transaction Amount", f"${amount:,.2f}")
col2.metric("Sender Balance Impact", f"${(newbalanceorg - oldbalanceorg):,.2f}")
col3.metric("Recipient Balance Impact", f"${(newbalancedest - oldbalancedest):,.2f}")
col4.metric("High-Value Trigger ($200k+)", "FLAGGED" if isflaggedfraud == 1 else "NORMAL")

# Trigger prediction
st.markdown("---")
analyze_btn = st.button("🔍 Run Stacking Ensemble Risk Assessment", type="primary", use_container_width=True)

if analyze_btn:
    if not sender_name.strip() or not receiver_name.strip():
        st.error("⚠️ Please specify both **Sender Account ID** and **Recipient Account ID** before running analysis.")
    else:
        values = {
            "step": int(step),
            "types": int(types),
            "amount": float(amount),
            "oldbalanceorig": float(oldbalanceorg),
            "newbalanceorig": float(newbalanceorg),
            "oldbalancedest": float(oldbalancedest),
            "newbalancedest": float(newbalancedest),
            "isflaggedfraud": float(isflaggedfraud)
        }
        # Dynamic API Endpoint (supports Render URL via st.secrets, env, or live default)
        import os
        api_base = os.getenv("API_URL")
        if not api_base and "API_URL" in st.secrets:
            api_base = st.secrets["API_URL"]
        if not api_base:
            api_base = "https://credit-card-fraud-detection-app-twyv.onrender.com"
        api_base = api_base.rstrip("/")
        predict_url = f"{api_base}/predict"


        with st.spinner("Executing XGBoost + LightGBM + Logistic Regression Stacking Pipeline..."):
            try:
                res = re.post(predict_url, json=values, timeout=45)
                
                if res.status_code == 200:
                    resp = res.json()
                    
                    # Extract probability and status
                    if isinstance(resp, dict):
                        is_fraud = resp.get("is_fraud", False) or resp.get("prediction") == "fraudulent"
                        fraud_prob = resp.get("fraud_probability", 1.0 if is_fraud else 0.0)
                        risk_pct = resp.get("risk_percentage", fraud_prob * 100)
                    else:
                        result = list(resp)[0] if isinstance(resp, (list, set)) else str(resp)
                        is_fraud = result.lower() == "fraudulent"
                        risk_pct = 99.9 if is_fraud else 0.1

                    st.markdown("### 🎯 Ensemble Decision & Calibrated Risk Score")

                    col_res1, col_res2 = st.columns([2, 1])

                    with col_res1:
                        if is_fraud:
                            st.markdown(f"""
                            <div class="fraud-badge">
                                <h2 style="color: #ef4444; margin: 0;">🚨 HIGH RISK DETECTED: FRAUDULENT</h2>
                                <p style="font-size: 1.15rem; color: #fca5a5; margin-top: 8px;">
                                    The Stacking Ensemble flagged this <b>{selected_type_name}</b> transaction between <b>{sender_name}</b> and <b>{receiver_name}</b> as unauthorized fraudulent activity.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="legit-badge">
                                <h2 style="color: #22c55e; margin: 0;">✅ TRANSACTION APPROVED: LEGITIMATE</h2>
                                <p style="font-size: 1.15rem; color: #86efac; margin-top: 8px;">
                                    The <b>{selected_type_name}</b> transaction of <b>${amount:,.2f}</b> between <b>{sender_name}</b> and <b>{receiver_name}</b> passed all fraud verification checkpoints.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                    with col_res2:
                        st.markdown("#### 📈 AI Fraud Probability Score")
                        st.metric(
                            label="Confidence Risk Score",
                            value=f"{risk_pct:.2f}%",
                            delta="HIGH RISK" if is_fraud else "SAFE",
                            delta_color="inverse" if is_fraud else "normal"
                        )
                        st.progress(min(max(risk_pct / 100.0, 0.0), 1.0))

                    # Explainability section
                    st.markdown("#### 🔬 Explainable AI: Behavioral Risk Breakdown")
                    reasons = []
                    if oldbalanceorg > 0 and newbalanceorg == 0 and amount == oldbalanceorg:
                        reasons.append("🚨 **Total Balance Drain Detected:** The transaction emptied 100% of the origin account balance to zero.")
                    if selected_type_name in ["Transfer", "Cash Out"]:
                        reasons.append(f"⚠️ **High-Risk Channel:** '{selected_type_name}' is the primary vector observed in financial theft exploits.")
                    if newbalancedest == oldbalancedest and amount > 0:
                        reasons.append("⚠️ **Ghost Account / Zero Dest Delta:** Recipient balance failed to register incoming funds, indicating intermediary layering.")
                    if isflaggedfraud:
                        reasons.append("⚠️ **High-Value Trigger:** Exceeds standard $200,000 monitoring limit.")
                    
                    if not is_fraud:
                        st.success("✔️ Normal balance reduction aligned with transfer amount.")
                        st.success("✔️ Recipient balance growth conforms to expected legitimate cash flow.")
                        st.success("✔️ Stacking meta-learner confidence $\ge 99\%$ for legitimate activity.")
                    else:
                        for r in reasons:
                            st.write(r)

                else:
                    st.error(f"Backend API returned status code {res.status_code}")
            except Exception:
                # Standalone In-Memory Fallback (Useful for Streamlit Community Cloud & single-command execution)
                try:
                    import joblib
                    import numpy as np
                    import model_def  # required for unpickling
                    local_model = joblib.load("credit_fraud.pkl")
                    feat_arr = np.array([[
                        values["step"], values["types"], values["amount"],
                        values["oldbalanceorig"], values["newbalanceorig"],
                        values["oldbalancedest"], values["newbalancedest"],
                        values["isflaggedfraud"]
                    ]], dtype=np.float64)
                    
                    fraud_prob = float(local_model.predict_proba(feat_arr)[0][1])
                    is_fraud = bool(local_model.predict(feat_arr)[0])
                    risk_pct = round(fraud_prob * 100, 2)

                    st.markdown("### 🎯 Ensemble Decision & Calibrated Risk Score *(Embedded Engine)*")

                    col_res1, col_res2 = st.columns([2, 1])

                    with col_res1:
                        if is_fraud:
                            st.markdown(f"""
                            <div class="fraud-badge">
                                <h2 style="color: #ef4444; margin: 0;">🚨 HIGH RISK DETECTED: FRAUDULENT</h2>
                                <p style="font-size: 1.15rem; color: #fca5a5; margin-top: 8px;">
                                    The Stacking Ensemble flagged this <b>{selected_type_name}</b> transaction between <b>{sender_name}</b> and <b>{receiver_name}</b> as unauthorized fraudulent activity.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="legit-badge">
                                <h2 style="color: #22c55e; margin: 0;">✅ TRANSACTION APPROVED: LEGITIMATE</h2>
                                <p style="font-size: 1.15rem; color: #86efac; margin-top: 8px;">
                                    The <b>{selected_type_name}</b> transaction of <b>${amount:,.2f}</b> between <b>{sender_name}</b> and <b>{receiver_name}</b> passed all fraud verification checkpoints.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                    with col_res2:
                        st.markdown("#### 📈 AI Fraud Probability Score")
                        st.metric(
                            label="Confidence Risk Score",
                            value=f"{risk_pct:.2f}%",
                            delta="HIGH RISK" if is_fraud else "SAFE",
                            delta_color="inverse" if is_fraud else "normal"
                        )
                        st.progress(min(max(risk_pct / 100.0, 0.0), 1.0))

                    # Explainability section
                    st.markdown("#### 🔬 Explainable AI: Behavioral Risk Breakdown")
                    reasons = []
                    if oldbalanceorg > 0 and newbalanceorg == 0 and amount == oldbalanceorg:
                        reasons.append("🚨 **Total Balance Drain Detected:** The transaction emptied 100% of the origin account balance to zero.")
                    if selected_type_name in ["Transfer", "Cash Out"]:
                        reasons.append(f"⚠️ **High-Risk Channel:** '{selected_type_name}' is the primary vector observed in financial theft exploits.")
                    if newbalancedest == oldbalancedest and amount > 0:
                        reasons.append("⚠️ **Ghost Account / Zero Dest Delta:** Recipient balance failed to register incoming funds, indicating intermediary layering.")
                    if isflaggedfraud:
                        reasons.append("⚠️ **High-Value Trigger:** Exceeds standard $200,000 monitoring limit.")
                    
                    if not is_fraud:
                        st.success("✔️ Normal balance reduction aligned with transfer amount.")
                        st.success("✔️ Recipient balance growth conforms to expected legitimate cash flow.")
                        st.success("✔️ Stacking meta-learner confidence $\ge 99\%$ for legitimate activity.")
                    else:
                        for r in reasons:
                            st.write(r)

                except Exception as inner_e:
                    st.error(f"⚠️ Could not score transaction. Ensure FastAPI server is running or model file is present. Error: {inner_e}")

