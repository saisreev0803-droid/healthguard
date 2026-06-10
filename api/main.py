import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import duckdb

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import PatientFeatures, PredictionResponse

# ─────────────────────────────────────────────────────────────────
# CREATE THE FASTAPI APP
# ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="HealthGuard API",
    description="Predicts 30-day hospital readmission risk using patient clinical features.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────
# LOAD MODEL AT STARTUP
# ─────────────────────────────────────────────────────────────────

print("Loading model...")

MODEL_PATH = "models/readmission_model.json"
model      = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

explainer = shap.TreeExplainer(model)

# Get the exact feature order the model was trained with
FEATURE_COLUMNS = model.get_booster().feature_names

print(f"Model loaded. Features: {FEATURE_COLUMNS}")

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def get_risk_label(score: float) -> str:
    if score >= 0.65:
        return "High Risk"
    elif score >= 0.35:
        return "Medium Risk"
    else:
        return "Low Risk"


def get_recommendation(label: str) -> str:
    recommendations = {
        "High Risk":   "Schedule follow-up within 7 days. Consider care coordinator referral and medication review before discharge.",
        "Medium Risk": "Schedule follow-up within 14 days. Provide discharge education and confirm patient has primary care access.",
        "Low Risk":    "Standard discharge protocol. Routine follow-up within 30 days.",
    }
    return recommendations[label]


# ─────────────────────────────────────────────────────────────────
# ENDPOINT 1 — Health check
# ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status":   "running",
        "model":    "readmission_xgboost_v1",
        "version":  "1.0.0",
        "features": len(FEATURE_COLUMNS),
    }


# ─────────────────────────────────────────────────────────────────
# ENDPOINT 2 — Predict readmission risk
# ─────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures):

    # Build input DataFrame using the exact feature order from the model
    patient_dict = patient.dict()

    input_data = pd.DataFrame(
        [[patient_dict[col] for col in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )

    # Get readmission probability
    risk_score = float(model.predict_proba(input_data)[0][1])

    # Calculate SHAP values
    shap_values = explainer.shap_values(input_data)[0]

    # Build top risk factors sorted by absolute impact
    shap_pairs = list(zip(FEATURE_COLUMNS, shap_values))
    shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    top_risk_factors = [
        {
            "feature":   name,
            "value":     float(input_data[name].iloc[0]),
            "impact":    round(float(impact), 4),
            "direction": "increases risk" if impact > 0 else "decreases risk",
        }
        for name, impact in shap_pairs[:5]
    ]

    risk_label     = get_risk_label(risk_score)
    recommendation = get_recommendation(risk_label)

    return PredictionResponse(
        risk_score       = round(risk_score, 4),
        risk_label       = risk_label,
        risk_percent     = f"{risk_score * 100:.1f}%",
        top_risk_factors = top_risk_factors,
        recommendation   = recommendation,
    )


# ─────────────────────────────────────────────────────────────────
# ENDPOINT 3 — Get patient by ID
# ─────────────────────────────────────────────────────────────────

@app.get("/patient/{patient_id}")
def get_patient(patient_id: str):
    try:
        con    = duckdb.connect("data/health.db")
        result = con.execute(
            "SELECT * FROM features WHERE patient_id = ?",
            [patient_id]
        ).df()
        con.close()

        if result.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Patient {patient_id} not found"
            )

        return result.to_dict(orient="records")[0]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────
# ENDPOINT 4 — High risk patients list
# ─────────────────────────────────────────────────────────────────

@app.get("/high-risk-patients")
def high_risk_patients(limit: int = 10):
    try:
        con    = duckdb.connect("data/health.db")
        result = con.execute(f"""
            SELECT
                f.patient_id,
                p.first_name || ' ' || p.last_name AS name,
                p.age,
                p.insurance,
                f.num_admissions_12m,
                f.num_chronic_conditions,
                f.has_heart_failure,
                f.has_diabetes,
                f.readmitted_30days
            FROM features f
            JOIN patients p ON f.patient_id = p.patient_id
            ORDER BY
                f.num_admissions_12m DESC,
                f.num_chronic_conditions DESC
            LIMIT {limit}
        """).df()
        con.close()

        return result.to_dict(orient="records")
# ─────────────────────────────────────────────────────────────────
# ENDPOINT 5 — Population analytics data
# ─────────────────────────────────────────────────────────────────

@app.get("/analytics")
def get_analytics():
    try:
        con = duckdb.connect("data/health.db")

        totals = con.execute("""
            SELECT
                COUNT(*)                                AS total_patients,
                ROUND(AVG(p.age), 1)                    AS avg_age,
                SUM(f.readmitted_30days)                AS total_readmitted,
                ROUND(AVG(f.num_chronic_conditions), 1) AS avg_conditions
            FROM features f
            JOIN patients p ON f.patient_id = p.patient_id
        """).df().iloc[0].to_dict()

        age_data = con.execute(
            "SELECT p.age FROM patients p"
        ).df()["age"].tolist()

        insurance_data = con.execute("""
            SELECT insurance, COUNT(*) AS count
            FROM patients GROUP BY insurance
        """).df().to_dict(orient="records")

        diagnosis_data = con.execute("""
            SELECT diagnosis, COUNT(*) AS total
            FROM conditions
            GROUP BY diagnosis
            ORDER BY total DESC LIMIT 10
        """).df().to_dict(orient="records")

        dept_data = con.execute("""
            SELECT
                department,
                COUNT(encounter_id)           AS total_visits,
                ROUND(AVG(length_of_stay), 1) AS avg_los
            FROM encounters
            GROUP BY department
            ORDER BY avg_los DESC
        """).df().to_dict(orient="records")

        con.close()

        return {
            "totals":       totals,
            "age_data":     age_data,
            "insurance":    insurance_data,
            "diagnoses":    diagnosis_data,
            "departments":  dept_data,
        }


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))