import duckdb
import pandas as pd

# ── Connect to your database ──────────────────────────────────────
con = duckdb.connect("data/health.db")
print("Connected to health.db")

# ─────────────────────────────────────────────────────────────────
# FEATURE 1 — Admission count and total length of stay (last 12 months)
# ─────────────────────────────────────────────────────────────────
# For every patient, count how many times they were admitted
# and how many total days they spent in hospital in the last year.

print("Building admission features...")

admissions = con.execute("""
    SELECT
        patient_id,
        COUNT(encounter_id)        AS num_admissions_12m,
        SUM(length_of_stay)        AS total_length_of_stay,
        MAX(length_of_stay)        AS max_single_stay,
        AVG(length_of_stay)        AS avg_length_of_stay
    FROM encounters
    WHERE admit_date >= CURRENT_DATE - INTERVAL 365 DAYS
    GROUP BY patient_id
""").df()

# ─────────────────────────────────────────────────────────────────
# FEATURE 2 — Chronic condition count and specific disease flags
# ─────────────────────────────────────────────────────────────────
# Count how many chronic conditions each patient has.
# Also create a 1/0 flag for the three highest-risk conditions.

print("Building condition features...")

condition_features = con.execute("""
    SELECT
        patient_id,
        SUM(CASE WHEN is_chronic = true THEN 1 ELSE 0 END)
            AS num_chronic_conditions,

        MAX(CASE WHEN diagnosis = 'Type 2 Diabetes Mellitus' THEN 1 ELSE 0 END)
            AS has_diabetes,

        MAX(CASE WHEN diagnosis = 'Heart Failure' THEN 1 ELSE 0 END)
            AS has_heart_failure,

        MAX(CASE WHEN diagnosis = 'Chronic Kidney Disease' THEN 1 ELSE 0 END)
            AS has_ckd,

        MAX(CASE WHEN diagnosis = 'Hypertension' THEN 1 ELSE 0 END)
            AS has_hypertension,

        MAX(CASE WHEN diagnosis = 'COPD' THEN 1 ELSE 0 END)
            AS has_copd
    FROM conditions
    GROUP BY patient_id
""").df()

# ─────────────────────────────────────────────────────────────────
# FEATURE 3 — Medication count
# ─────────────────────────────────────────────────────────────────
# Count how many active medications each patient currently has.
# Patients on many medications are medically complex.

print("Building medication features...")

medication_features = con.execute("""
    SELECT
        patient_id,
        COUNT(med_id)                                    AS num_medications,
        SUM(CASE WHEN is_active = true THEN 1 ELSE 0 END) AS num_active_medications
    FROM medications
    GROUP BY patient_id
""").df()

# ─────────────────────────────────────────────────────────────────
# FEATURE 4 — Lab result averages
# ─────────────────────────────────────────────────────────────────
# For the three most clinically important labs, calculate each
# patient's average value across all their recorded results.

print("Building lab features...")

lab_features = con.execute("""
    SELECT
        patient_id,
        ROUND(AVG(CASE WHEN test_name = 'Blood Glucose'  THEN value END), 2)
            AS avg_glucose,
        ROUND(AVG(CASE WHEN test_name = 'Systolic BP'    THEN value END), 2)
            AS avg_systolic_bp,
        ROUND(AVG(CASE WHEN test_name = 'Creatinine'     THEN value END), 2)
            AS avg_creatinine,
        ROUND(AVG(CASE WHEN test_name = 'HbA1c'          THEN value END), 2)
            AS avg_hba1c,
        ROUND(AVG(CASE WHEN test_name = 'Heart Rate'     THEN value END), 2)
            AS avg_heart_rate
    FROM observations
    GROUP BY patient_id
""").df()

# ─────────────────────────────────────────────────────────────────
# FEATURE 5 — Demographics
# ─────────────────────────────────────────────────────────────────
# Pull age and encode insurance type as a number.
# ML models need numbers, not text like "Medicare".

print("Building demographic features...")

demographics = con.execute("""
    SELECT
        patient_id,
        age,
        gender,
        CASE insurance
            WHEN 'Medicare'   THEN 0
            WHEN 'Medicaid'   THEN 1
            WHEN 'Private'    THEN 2
            WHEN 'Uninsured'  THEN 3
            ELSE 4
        END AS insurance_encoded,
        CASE gender
            WHEN 'Male'   THEN 0
            WHEN 'Female' THEN 1
            ELSE 2
        END AS gender_encoded
    FROM patients
""").df()

# ─────────────────────────────────────────────────────────────────
# FEATURE 6 — Readmission target label (what we're predicting)
# ─────────────────────────────────────────────────────────────────
# For each patient, check if any two encounters happened within
# 30 days of each other. If yes, label = 1 (readmitted). If no, 0.

print("Building readmission target label...")

readmission = con.execute("""
    SELECT DISTINCT
        e1.patient_id,
        1 AS readmitted_30days
    FROM encounters e1
    JOIN encounters e2
        ON  e1.patient_id   = e2.patient_id
        AND e1.encounter_id != e2.encounter_id
        AND e2.admit_date BETWEEN e1.discharge_date
                              AND e1.discharge_date + INTERVAL 30 DAYS
""").df()

# ─────────────────────────────────────────────────────────────────
# JOIN ALL FEATURES TOGETHER
# ─────────────────────────────────────────────────────────────────
# Start with all patients, then left-join every feature table.
# Left join means: keep ALL patients even if they have no value
# for a feature (we'll fill those gaps with 0 or a median).

print("Joining all features together...")

all_patients = con.execute("SELECT patient_id FROM patients").df()

features = (
    all_patients
    .merge(demographics,        on="patient_id", how="left")
    .merge(admissions,          on="patient_id", how="left")
    .merge(condition_features,  on="patient_id", how="left")
    .merge(medication_features, on="patient_id", how="left")
    .merge(lab_features,        on="patient_id", how="left")
    .merge(readmission,         on="patient_id", how="left")
)

# ─────────────────────────────────────────────────────────────────
# FILL MISSING VALUES
# ─────────────────────────────────────────────────────────────────
# Some patients may have no lab results or no encounters in the
# last 12 months. Fill those gaps sensibly:
# - Counts (admissions, conditions) → 0
# - Lab values → median of all patients (most neutral estimate)
# - Target label → 0 (not readmitted if no evidence of it)

print("Filling missing values...")

count_columns = [
    "num_admissions_12m", "total_length_of_stay", "max_single_stay",
    "avg_length_of_stay", "num_chronic_conditions", "has_diabetes",
    "has_heart_failure", "has_ckd", "has_hypertension", "has_copd",
    "num_medications", "num_active_medications",
]

lab_columns = [
    "avg_glucose", "avg_systolic_bp", "avg_creatinine",
    "avg_hba1c", "avg_heart_rate",
]

for col in count_columns:
    features[col] = features[col].fillna(0)

for col in lab_columns:
    features[col] = features[col].fillna(features[col].median())

features["readmitted_30days"] = features["readmitted_30days"].fillna(0).astype(int)

# ─────────────────────────────────────────────────────────────────
# SAVE THE FEATURE TABLE
# ─────────────────────────────────────────────────────────────────

features.to_csv("data/features.csv", index=False)

# Also save it back into DuckDB as a proper table
con.execute("CREATE OR REPLACE TABLE features AS SELECT * FROM features")

print("\nFeature engineering complete!")
print(f"  Total patients:       {len(features)}")
print(f"  Total features:       {len(features.columns) - 1}")
print(f"  Readmitted (label=1): {features['readmitted_30days'].sum()}")
print(f"  Not readmitted (0):   {(features['readmitted_30days'] == 0).sum()}")
print("\nSaved to: data/features.csv and health.db → features table")