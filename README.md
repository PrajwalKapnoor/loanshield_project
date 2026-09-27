# 🛡️ LoanShield — Lending Fraud & Default Detection Engine

> Can an ensemble model flag high-risk loan applicants before disbursement — and quantify exactly how much it saves the lender?

---

## Problem Statement
Banks and lending institutions lose billions to loan defaults annually.
Most ML models optimise for accuracy — but a False Negative (approving a bad loan) costs **~$18,500** in lost principal, while a False Positive (rejecting a good borrower) costs only **~$1,200** in missed interest. This project treats fraud detection as a **business cost minimisation problem**, not an accuracy contest.

---

## Dataset
- **Source:** [HMEQ — Home Equity Loan Default (Kaggle)](https://www.kaggle.com/datasets/ajay1735/hmeq-data)
- **Size:** 5,960 rows × 13 features
- **Target:** `BAD` — 1 = defaulted / severely delinquent, 0 = repaid
- **Fraud rate:** 19.9%
- **Note:** Raw data not committed. Download `hmeq.csv` from Kaggle and place in `data/` before running the notebook.

---

## Approach

**Act 1 — Fraud Landscape** (`notebooks/loanshield_fraud_detection.ipynb` → Cells 1–8)
- EDA: class imbalance, feature distributions, missing data audit
- Class-conditional median imputation — fills missing values separately per class to preserve fraud signal

**Act 2 — Ensemble Showdown** (Cells 9–12)
- Decision Tree baseline — exposes high-variance single-tree problem
- Random Forest (Bagging) — 200 bootstrap trees, √n feature subsets, variance reduction
- XGBoost (Boosting) — sequential residual correction, `scale_pos_weight` from class ratio

**Act 3 — Business Verdict** (Cells 13–15)
- Cost matrix: FN = $18,500 / FP = $1,200 (15:1 ratio)
- 90-point threshold sweep to find cost-minimising decision boundary
- Feature importance: RF (MDI) vs XGBoost (Gain) side-by-side

**Phase 1 — Serialisation** (Cell 15)
- Saves trained model + encoders + medians + metadata to `models/`

---

## Key Results

| Model | ROC-AUC | vs Baseline |
|-------|---------|-------------|
| Decision Tree (Baseline) | 0.9005 | — |
| **Random Forest (Bagging)** 🏆 | **0.9810** | **+0.0805** |
| XGBoost (Boosting) | 0.9756 | +0.0750 |

**Winner: Random Forest** — strong raw signals (DEBTINC, DEROG, DELINQ) meant bagging's variance reduction was the decisive factor over boosting's sequential correction.

---

## Project Structure

```
loanshield-fraud-detection/
├── app.py                    ← FastAPI backend (3 endpoints)
├── streamlit_app.py          ← Streamlit dashboard entry point
├── requirements.txt
├── Procfile                  ← Render deployment config
├── pages/
│   ├── 1_Loan_Screener.py         ← Live risk verdict form
│   ├── 2_Model_Intelligence.py    ← Model comparison charts
│   └── 3_Business_Cost_Dashboard.py ← Threshold + cost analysis
├── models/                   ← Paste .pkl files here (from Cell 15)
├── notebooks/
│   └── loanshield_fraud_detection.ipynb
├── outputs/figures/          ← Paste 3 PNG charts here (from notebook)
└── data/processed/           ← Paste hmeq_clean.csv here
```

---

## How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the notebook end-to-end → generates models/ and outputs/figures/

# 3. Start FastAPI backend (Terminal 1)
uvicorn app:app --reload

# 4. Start Streamlit dashboard (Terminal 2)
streamlit run streamlit_app.py
```

Open `http://localhost:8501` → navigate to Loan Screener → submit an application.

---

## Deployment

| Service | What | Command |
|---------|------|---------|
| **Render** | FastAPI API | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
| **Streamlit Cloud** | Dashboard | Set `API_URL` in Secrets |

---

## Tech Stack
`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `FastAPI` · `Pydantic v2` · `Streamlit` · `Matplotlib` · `Seaborn` · `Joblib`
