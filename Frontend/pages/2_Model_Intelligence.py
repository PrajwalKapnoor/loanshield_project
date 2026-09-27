# =============================================================================
#  LoanShield — Page 2: Model Intelligence
#  Loads 8 individual charts from outputs/figures/
#  Act 1: 4 fraud landscape charts
#  Act 2: 4 ensemble comparison charts
# =============================================================================

import streamlit as st
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="Model Intelligence", page_icon="⚔️", layout="wide")

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

st.title("⚔️ Ensemble Showdown — Model Intelligence")
st.markdown("Three models, one winner. Here's how each performed and why.")
st.markdown("---")

# ── Model scorecard metrics ───────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Decision Tree  (Baseline)",   "AUC 0.9005", "High variance — single tree")
c2.metric("Random Forest  (Bagging) 🏆", "AUC 0.9810", "+0.0805 vs baseline")
c3.metric("XGBoost        (Boosting)",   "AUC 0.9756", "+0.0750 vs baseline")

st.info(
    "**Why Random Forest beat XGBoost here:** "
    "The top fraud signals (DEBTINC, DEROG, DELINQ) are strong enough individually "
    "that bagging's variance reduction was the decisive factor — "
    "XGBoost's sequential correction adds no meaningful advantage "
    "when the underlying signals are already this clean."
)

st.markdown("---")

# ── Act 1: Fraud Landscape ────────────────────────────────────────────────────
st.markdown("## Act 1 — Fraud Landscape")
st.markdown("Understand the risk landscape before touching a single model.")
st.markdown("")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Class Distribution**")
    try:
        st.image(load_figure("act1_class_distribution.png"), use_container_width=True)
    except FileNotFoundError:
        st.warning("Run notebook to generate act1_class_distribution.png")

    st.markdown("**Derogatory Reports by Class**")
    try:
        st.image(load_figure("act1_derogatory_reports.png"), use_container_width=True)
    except FileNotFoundError:
        st.warning("Run notebook to generate act1_derogatory_reports.png")

with col2:
    st.markdown("**Debt-to-Income Distribution**")
    try:
        st.image(load_figure("act1_debt_to_income.png"), use_container_width=True)
    except FileNotFoundError:
        st.warning("Run notebook to generate act1_debt_to_income.png")

    st.markdown("**Missing Data % by Class**")
    try:
        st.image(load_figure("act1_missing_data.png"), use_container_width=True)
    except FileNotFoundError:
        st.warning("Run notebook to generate act1_missing_data.png")

st.markdown("---")

# ── Act 2: Ensemble Showdown ──────────────────────────────────────────────────
st.markdown("## Act 2 — Ensemble Showdown")
st.markdown("ROC curves, Precision-Recall curves, cross-validation variance, and confusion matrix.")
st.markdown("")

col3, col4 = st.columns(2)

with col3:
    st.markdown("**ROC Curves — All 3 Models**")
    try:
        st.image(load_figure("act2_roc_curves.png"), use_container_width=True)
    except FileNotFoundError:
        st.warning("Run notebook to generate act2_roc_curves.png")

    st.markdown("**CV Score Distribution — Variance Comparison**")
    try:
        st.image(load_figure("act2_cv_scores.png"), use_container_width=True)
    except FileNotFoundError:
        st.warning("Run notebook to generate act2_cv_scores.png")

with col4:
    st.markdown("**Precision-Recall Curves — Fraud Class Focus**")
    try:
        st.image(load_figure("act2_pr_curves.png"), use_container_width=True)
    except FileNotFoundError:
        st.warning("Run notebook to generate act2_pr_curves.png")

    st.markdown("**Confusion Matrix — Best Model**")
    try:
        st.image(load_figure("act2_confusion_matrix.png"), use_container_width=True)
    except FileNotFoundError:
        st.warning("Run notebook to generate act2_confusion_matrix.png")
