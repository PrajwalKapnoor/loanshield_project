# =============================================================================
#  LoanShield — Streamlit Dashboard Entry Point
#  Navigate via sidebar to:
#    Page 1 — Live Loan Screener
#    Page 2 — Model Intelligence
#    Page 3 — Business Cost Dashboard
# =============================================================================

import streamlit as st

st.set_page_config(
    page_title = "LoanShield",
    page_icon  = "🛡️",
    layout     = "wide"
)

st.title("🛡️ LoanShield")
st.subheader("Lending Fraud & Default Detection Engine")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔍 Loan Screener")
    st.markdown(
        "Submit a live loan application and get an instant risk verdict — "
        "fraud probability, risk tier, decision, and expected cost at risk."
    )

with col2:
    st.markdown("### ⚔️ Model Intelligence")
    st.markdown(
        "See how the three ensemble models compare — "
        "Decision Tree vs Random Forest vs XGBoost — "
        "across ROC-AUC, Precision-Recall, and cross-validation variance."
    )

with col3:
    st.markdown("### 💰 Business Cost Dashboard")
    st.markdown(
        "Understand the financial impact. Threshold optimisation, "
        "cost matrix breakdown, and feature importance across "
        "both ensemble families."
    )

st.markdown("---")
st.markdown(
    "**Dataset:** HMEQ — Home Equity Loan Default &nbsp;|&nbsp; "
    "**Rows:** 5,960 &nbsp;|&nbsp; "
    "**Fraud rate:** 19.9% &nbsp;|&nbsp; "
    "**Winner:** Random Forest (AUC 0.9810)"
)
