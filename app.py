# =============================================================================
#  LoanShield — FastAPI Backend
#  Serves the trained Random Forest model via three endpoints:
#    GET  /health           — liveness check
#    GET  /threshold-info   — returns optimised threshold + cost params
#    POST /predict          — accepts a loan application, returns risk verdict
# =============================================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional
import joblib
import numpy as np

from pathlib import Path

# ── Model store (loaded once at startup, reused for every request) ────────────
models = {}

# Locate models directory relative to project root or CWD
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" if (BASE_DIR / "models").exists() else Path("models")

@asynccontextmanager
async def lifespan(app: FastAPI):
    models["rf"]                = joblib.load(MODELS_DIR / "random_forest_model.pkl")
    models["le_reason"]         = joblib.load(MODELS_DIR / "label_encoder_reason.pkl")
    models["le_job"]            = joblib.load(MODELS_DIR / "label_encoder_job.pkl")
    models["inference_medians"] = joblib.load(MODELS_DIR / "inference_medians.pkl")
    models["metadata"]          = joblib.load(MODELS_DIR / "model_metadata.pkl")
    print(f"✓ All model artifacts loaded from {MODELS_DIR}")
    yield
    models.clear()

app = FastAPI(
    title       = "LoanShield API",
    description = "Lending Fraud & Default Detection — Random Forest (AUC 0.9810)",
    version     = "1.0.0",
    lifespan    = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Input schema ──────────────────────────────────────────────────────────────
# All fields except LOAN are Optional — missing values are filled using
# inference_medians (overall training medians frozen at serialisation time).
class LoanApplication(BaseModel):
    LOAN:    float
    MORTDUE: Optional[float] = None
    VALUE:   Optional[float] = None
    REASON:  str             = "DebtCon"
    JOB:     str             = "Other"
    YOJ:     Optional[float] = None
    DEROG:   Optional[float] = None
    DELINQ:  Optional[float] = None
    CLAGE:   Optional[float] = None
    NINQ:    Optional[float] = None
    CLNO:    Optional[float] = None
    DEBTINC: Optional[float] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model":  "Random Forest",
        "auc":    0.9810
    }


@app.get("/threshold-info")
def threshold_info():
    meta = models["metadata"]
    return {
        "opt_threshold":   meta["opt_threshold"],
        "avg_loan_amount": meta["avg_loan_amount"],
        "annual_interest": meta["annual_interest"],
        "fraud_rate":      meta["fraud_rate"],
    }


@app.post("/predict")
def predict(loan: LoanApplication):
    data    = loan.model_dump()
    medians = models["inference_medians"]
    meta    = models["metadata"]

    # ── Step 1: Impute missing numeric fields ─────────────────────────────────
    # Uses overall training medians — consistent with how the model was trained.
    for col, median_val in medians.items():
        if data.get(col) is None:
            data[col] = median_val

    # ── Step 2: Encode categorical features ──────────────────────────────────
    try:
        reason_enc = int(models["le_reason"].transform([data["REASON"]])[0])
    except ValueError:
        reason_enc = 0   # unseen category → fall back to first class

    try:
        job_enc = int(models["le_job"].transform([data["JOB"]])[0])
    except ValueError:
        job_enc = 0

    # ── Step 3: Build feature vector in exact training order ──────────────────
    # FEATURE_COLS = ["LOAN","MORTDUE","VALUE","YOJ","DEROG","DELINQ",
    #                 "CLAGE","NINQ","CLNO","DEBTINC","REASON_ENC","JOB_ENC"]
    feature_vector = np.array([[
        data["LOAN"],    data["MORTDUE"], data["VALUE"],
        data["YOJ"],     data["DEROG"],   data["DELINQ"],
        data["CLAGE"],   data["NINQ"],    data["CLNO"],
        data["DEBTINC"], reason_enc,      job_enc
    ]])

    # ── Step 4: Predict ───────────────────────────────────────────────────────
    fraud_prob = float(models["rf"].predict_proba(feature_vector)[0][1])
    threshold  = meta["opt_threshold"]

    # ── Step 5: Risk tier ─────────────────────────────────────────────────────
    if fraud_prob >= 0.65:
        risk_tier = "HIGH"
    elif fraud_prob >= 0.35:
        risk_tier = "MEDIUM"
    else:
        risk_tier = "LOW"

    # ── Step 6: Decision ──────────────────────────────────────────────────────
    if fraud_prob >= threshold:
        decision = "REJECT"
    elif fraud_prob >= threshold * 0.75:
        decision = "MANUAL REVIEW"
    else:
        decision = "APPROVE"

    # ── Step 7: Business cost at risk ─────────────────────────────────────────
    # Expected loss = fraud_probability × average loan principal
    cost_at_risk = round(fraud_prob * meta["avg_loan_amount"], 2)

    # ── Step 8: Top 3 risk factors by model feature importance ───────────────
    importances    = models["rf"].feature_importances_
    feature_names  = meta["feature_cols"]
    top_3_indices  = np.argsort(importances)[::-1][:3]
    top_3_factors  = [feature_names[i] for i in top_3_indices]

    return {
        "fraud_probability": round(fraud_prob, 4),
        "risk_tier":         risk_tier,
        "decision":          decision,
        "cost_at_risk":      cost_at_risk,
        "threshold_used":    round(threshold, 2),
        "top_risk_factors":  top_3_factors
    }
