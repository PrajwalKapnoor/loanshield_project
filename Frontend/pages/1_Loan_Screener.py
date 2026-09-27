# =============================================================================
#  LoanShield — Page 1: Live Loan Screener
#  Calls POST /predict on the FastAPI backend.
#  Returns: fraud probability, risk tier, decision, cost at risk.
# =============================================================================

import streamlit as st
import requests

st.set_page_config(
    page_title = "Loan Screener",
    page_icon  = "🔍",
    layout     = "wide"
)

# ── API URL ───────────────────────────────────────────────────────────────────
# Local dev  : http://localhost:8000
# Deployed   : set API_URL in Streamlit Cloud → Settings → Secrets
try:
    API_URL = st.secrets.get("API_URL", "http://localhost:8000")
except Exception:
    API_URL = "http://localhost:8000"

st.title("🔍 Live Loan Risk Screener")
st.markdown("Fill in the application details and click **Assess Risk** for an instant verdict.")
st.markdown("---")

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("loan_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Loan Details**")
        LOAN    = st.number_input("Loan Amount ($)",         min_value=1000,  max_value=100000, value=18000,  step=500)
        MORTDUE = st.number_input("Mortgage Balance Due ($)", min_value=0,     max_value=500000, value=73000,  step=1000)
        VALUE   = st.number_input("Property Value ($)",      min_value=0,     max_value=999999, value=101000, step=1000)
        DEBTINC = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0,  max_value=100.0,  value=35.0,   step=0.5)

    with col2:
        st.markdown("**Borrower Profile**")
        REASON = st.selectbox("Loan Reason",  ["DebtCon", "HomeImp"])
        JOB    = st.selectbox("Job Category", ["Other", "ProfExe", "Office", "Mgr", "Self", "Sales"])
        YOJ    = st.number_input("Years at Current Job",         min_value=0.0, max_value=40.0,  value=7.0,   step=0.5)
        CLAGE  = st.number_input("Age of Oldest Credit Line (months)", min_value=0.0, max_value=600.0, value=180.0, step=6.0)

    with col3:
        st.markdown("**Credit History**")
        DEROG  = st.number_input("Derogatory Reports",       min_value=0, max_value=20, value=0)
        DELINQ = st.number_input("Delinquent Credit Lines",  min_value=0, max_value=20, value=0)
        NINQ   = st.number_input("Recent Credit Inquiries",  min_value=0, max_value=20, value=1)
        CLNO   = st.number_input("Total Credit Lines",       min_value=0, max_value=100, value=20)

    submitted = st.form_submit_button("🔍 Assess Risk", use_container_width=True)

# ── Result display ────────────────────────────────────────────────────────────
if submitted:
    payload = {
        "LOAN": LOAN, "MORTDUE": MORTDUE, "VALUE": VALUE,
        "REASON": REASON, "JOB": JOB, "YOJ": YOJ,
        "DEROG": DEROG, "DELINQ": DELINQ, "CLAGE": CLAGE,
        "NINQ": NINQ, "CLNO": CLNO, "DEBTINC": DEBTINC
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        result   = response.json()

        st.markdown("---")
        st.markdown("### Risk Verdict")

        # ── Colour-coded tier badge ────────────────────────────────────────────
        tier   = result["risk_tier"]
        colors = {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}
        icons  = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}

        st.markdown(
            f"<div style='background:{colors[tier]};padding:12px 20px;"
            f"border-radius:8px;display:inline-block;'>"
            f"<b style='color:white;font-size:20px'>{icons[tier]}  {tier} RISK</b>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown("")

        # ── Key metrics ──────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fraud Probability",  f"{result['fraud_probability'] * 100:.1f}%")
        m2.metric("Decision",           result["decision"])
        m3.metric("Cost at Risk",       f"${result['cost_at_risk']:,.0f}")
        m4.metric("Threshold Used",     result["threshold_used"])

        # ── Top risk factors ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**Top 3 risk factors driving this prediction:**")
        factor_labels = {
            "DEBTINC":     "Debt-to-Income Ratio",
            "DEROG":       "Derogatory Reports",
            "DELINQ":      "Delinquent Credit Lines",
            "CLAGE":       "Age of Oldest Credit Line",
            "VALUE":       "Property Value",
            "LOAN":        "Loan Amount",
            "MORTDUE":     "Mortgage Balance",
            "YOJ":         "Years at Job",
            "NINQ":        "Recent Inquiries",
            "CLNO":        "Total Credit Lines",
            "REASON_ENC":  "Loan Reason",
            "JOB_ENC":     "Job Category",
        }
        for i, factor in enumerate(result["top_risk_factors"], 1):
            label = factor_labels.get(factor, factor)
            st.markdown(f"  **{i}.** {label} (`{factor}`)")

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the LoanShield API. "
            "Make sure the FastAPI server is running on port 8000 "
            "(`uvicorn app:app --reload`)."
        )
    except Exception as e:
        st.error(f"Unexpected error: {e}")
