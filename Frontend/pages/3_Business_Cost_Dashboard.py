# =============================================================================
#  LoanShield — Page 3: Business Cost Dashboard
#  Loads 4 individual Act 3 charts from outputs/figures/
# =============================================================================

import streamlit as st
import requests
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="Business Cost Dashboard", page_icon="💰", layout="wide")

try:
    API_URL = st.secrets.get("API_URL", "http://localhost:8000")
except Exception:
    API_URL = "http://localhost:8000"

def load_figure(filename: str):
    p1 = Path("outputs/figures") / filename
    if p1.exists():
        return Image.open(p1)
    p2 = Path(__file__).resolve().parents[2] / "outputs" / "figures" / filename
    if p2.exists():
        return Image.open(p2)
    p3 = Path(__file__).resolve().parents[1] / "outputs" / "figures" / filename
    if p3.exists():
        return Image.open(p3)
    raise FileNotFoundError(f"Figure {filename} not found")

st.title("💰 Business Cost Dashboard")
st.markdown("Model performance translated into dollars. The part most ML projects skip.")
st.markdown("---")

# ── Cost matrix explanation ───────────────────────────────────────────────────
st.markdown("### Why Accuracy Is the Wrong Metric for a Lender")

col1, col2, col3 = st.columns(3)

col1.markdown(
    "<div style='background:#e74c3c22;border-left:4px solid #e74c3c;"
    "padding:16px;border-radius:4px'>"
    "<b style='color:#e74c3c'>❌ False Negative</b><br><br>"
    "Model says <b>APPROVE</b><br>"
    "Borrower defaults<br><br>"
    "<b>Cost: ~$18,500</b><br>"
    "<small>Full loan principal lost</small>"
    "</div>",
    unsafe_allow_html=True
)

col2.markdown(
    "<div style='background:#f39c1222;border-left:4px solid #f39c12;"
    "padding:16px;border-radius:4px'>"
    "<b style='color:#f39c12'>⚠️ False Positive</b><br><br>"
    "Model says <b>REJECT</b><br>"
    "Borrower was legit<br><br>"
    "<b>Cost: ~$1,200</b><br>"
    "<small>One year of interest lost</small>"
    "</div>",
    unsafe_allow_html=True
)

col3.markdown(
    "<div style='background:#2ecc7122;border-left:4px solid #2ecc71;"
    "padding:16px;border-radius:4px'>"
    "<b style='color:#2ecc71'>📊 Cost Ratio</b><br><br>"
    "FN is <b>15×</b> more<br>expensive than FP<br><br>"
    "<b>Implication</b><br>"
    "<small>Default threshold 0.50 is wrong.<br>Tune it to minimise total cost.</small>"
    "</div>",
    unsafe_allow_html=True
)

st.markdown("")
st.markdown("---")

# ── Live threshold info from API ──────────────────────────────────────────────
st.markdown("### Live Threshold Info")
try:
    info = requests.get(f"{API_URL}/threshold-info", timeout=5).json()
    t1, t2, t3 = st.columns(3)
    t1.metric("Optimised Threshold",  info["opt_threshold"],                "vs default 0.50")
    t2.metric("Avg Loan Amount",      f"${info['avg_loan_amount']:,.0f}",   "FN cost basis")
    t3.metric("Dataset Fraud Rate",   f"{info['fraud_rate']*100:.1f}%",     "Class imbalance")
except Exception:
    st.warning("API not reachable — start the FastAPI server to see live threshold values.")

st.markdown("---")

# ── Act 3 charts — 4 individual ───────────────────────────────────────────────
st.markdown("## Act 3 — Business Verdict Charts")
st.markdown("")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Threshold Optimisation — Minimise Loan Losses**")
    try:
        st.image(load_figure("act3_threshold_cost.png"), use_container_width=True)
    except FileNotFoundError:
        st.warning("Run notebook to generate act3_threshold_cost.png")

    st.markdown("**Feature Importance — Random Forest vs XGBoost**")
    try:
        st.image(load_figure("act3_feature_importance.png"), use_container_width=True)
    except FileNotFoundError:
        st.warning("Run notebook to generate act3_feature_importance.png")

with col_b:
    st.markdown("**Precision vs Recall Trade-off by Threshold**")
    try:
        st.image(load_figure("act3_precision_recall_tradeoff.png"), use_container_width=True)
    except FileNotFoundError:
        st.warning("Run notebook to generate act3_precision_recall_tradeoff.png")

    st.markdown("**Final Model Scorecard — Business Metrics**")
    try:
        st.image(load_figure("act3_model_scorecard.png"), use_container_width=True)
    except FileNotFoundError:
        st.warning("Run notebook to generate act3_model_scorecard.png")

st.markdown("---")
st.info(
    "**Key takeaway:** The optimised threshold was found by computing "
    "(FN × $18,500) + (FP × $1,200) across 90 threshold values. "
    "The minimum-cost point is used as the decision boundary in the Live Loan Screener."
)
