from pydantic import BaseModel, Field
from typing import Optional

# ─────────────────────────────────────────────────────────────────
# INPUT SCHEMA — what the caller must send to /predict
# ─────────────────────────────────────────────────────────────────
# Each field has:
#   - a type (float, int)
#   - a description (shown in the auto-generated docs)
#   - ge/le constraints (ge = greater or equal, le = less or equal)
# If the caller sends a value outside these bounds, FastAPI
# automatically rejects the request with a clear error message.

class PatientFeatures(BaseModel):

    age:                    float = Field(..., ge=18,  le=110,  description="Patient age in years")
    gender_encoded:         int   = Field(..., ge=0,   le=2,    description="0=Male 1=Female 2=Other")
    insurance_encoded:      int   = Field(..., ge=0,   le=4,    description="0=Medicare 1=Medicaid 2=Private 3=Uninsured")
    num_admissions_12m:     float = Field(..., ge=0,            description="Hospital admissions in last 12 months")
    total_length_of_stay:   float = Field(..., ge=0,            description="Total days in hospital last 12 months")
    max_single_stay:        float = Field(..., ge=0,            description="Longest single hospital stay in days")
    avg_length_of_stay:     float = Field(..., ge=0,            description="Average length of stay in days")
    num_chronic_conditions: float = Field(..., ge=0,   le=15,   description="Number of chronic conditions")
    has_diabetes:           int   = Field(..., ge=0,   le=1,    description="1 if diagnosed with diabetes")
    has_heart_failure:      int   = Field(..., ge=0,   le=1,    description="1 if diagnosed with heart failure")
    has_ckd:                int   = Field(..., ge=0,   le=1,    description="1 if diagnosed with chronic kidney disease")
    has_hypertension:       int   = Field(..., ge=0,   le=1,    description="1 if diagnosed with hypertension")
    has_copd:               int   = Field(..., ge=0,   le=1,    description="1 if diagnosed with COPD")
    num_medications:        float = Field(..., ge=0,            description="Total number of medications")
    num_active_medications: float = Field(..., ge=0,            description="Number of currently active medications")
    avg_glucose:            float = Field(..., ge=0,            description="Average blood glucose mg/dL")
    avg_systolic_bp:        float = Field(..., ge=0,            description="Average systolic blood pressure mmHg")
    avg_creatinine:         float = Field(..., ge=0,            description="Average creatinine level mg/dL")
    avg_hba1c:              float = Field(..., ge=0,            description="Average HbA1c percentage")
    avg_heart_rate:         float = Field(..., ge=0,            description="Average heart rate bpm")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 67,
                "gender_encoded": 0,
                "insurance_encoded": 0,
                "num_admissions_12m": 3,
                "total_length_of_stay": 18,
                "max_single_stay": 9,
                "avg_length_of_stay": 6.0,
                "num_chronic_conditions": 4,
                "has_diabetes": 1,
                "has_heart_failure": 1,
                "has_ckd": 0,
                "has_hypertension": 1,
                "has_copd": 0,
                "num_medications": 7,
                "num_active_medications": 5,
                "avg_glucose": 210.5,
                "avg_systolic_bp": 148.0,
                "avg_creatinine": 1.8,
                "avg_hba1c": 8.2,
                "avg_heart_rate": 88.0
            }
        }


# ─────────────────────────────────────────────────────────────────
# OUTPUT SCHEMA — what /predict sends back to the caller
# ─────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    risk_score:       float       # probability 0.0 to 1.0
    risk_label:       str         # "Low Risk" / "Medium Risk" / "High Risk"
    risk_percent:     str         # human readable e.g. "73.4%"
    top_risk_factors: list        # top SHAP features driving the prediction
    recommendation:   str         # plain English clinical suggestion