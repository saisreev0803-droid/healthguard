import random
import uuid
import pandas as pd
from faker import Faker
from datetime import timedelta

fake = Faker()
random.seed(42)
Faker.seed(42)

NUM_PATIENTS = 1000

# ── Lists of realistic clinical values ───────────────────────────

DIAGNOSES = [
    "Type 2 Diabetes Mellitus",
    "Hypertension",
    "Coronary Artery Disease",
    "Chronic Kidney Disease",
    "Heart Failure",
    "COPD",
    "Atrial Fibrillation",
    "Pneumonia",
    "Urinary Tract Infection",
    "Sepsis",
    "Stroke",
    "Anemia",
    "Obesity",
    "Asthma",
    "Depression",
]

MEDICATIONS = [
    "Metformin 500mg",
    "Lisinopril 10mg",
    "Atorvastatin 20mg",
    "Aspirin 81mg",
    "Furosemide 40mg",
    "Amlodipine 5mg",
    "Omeprazole 20mg",
    "Albuterol inhaler",
    "Warfarin 5mg",
    "Insulin Glargine",
]

LAB_TESTS = [
    ("HbA1c",           4.0,  13.0, "%"),
    ("Blood Glucose",   60,   400,  "mg/dL"),
    ("Creatinine",      0.5,  8.0,  "mg/dL"),
    ("Hemoglobin",      6.0,  18.0, "g/dL"),
    ("Systolic BP",     90,   200,  "mmHg"),
    ("Diastolic BP",    50,   130,  "mmHg"),
    ("Heart Rate",      45,   130,  "bpm"),
    ("WBC Count",       2.0,  20.0, "K/uL"),
    ("Sodium",          125,  155,  "mEq/L"),
    ("Potassium",       2.5,  6.5,  "mEq/L"),
]

ENCOUNTER_TYPES = [
    "Emergency Visit",
    "Inpatient Admission",
    "Outpatient Visit",
    "ICU Admission",
]

# ─────────────────────────────────────────────────────────────────
# GENERATE PATIENTS
# ─────────────────────────────────────────────────────────────────

patients     = []
encounters   = []
conditions   = []
observations = []
medications  = []

print("Generating patients...")

for i in range(NUM_PATIENTS):

    patient_id  = str(uuid.uuid4())
    birth_date  = fake.date_of_birth(minimum_age=18, maximum_age=90)
    age         = (pd.Timestamp.today().date() - birth_date).days // 365

    patient = {
        "patient_id":  patient_id,
        "first_name":  fake.first_name(),
        "last_name":   fake.last_name(),
        "gender":      random.choice(["Male", "Female"]),
        "birth_date":  birth_date,
        "age":         age,
        "city":        fake.city(),
        "state":       fake.state(),
        "postal_code": fake.zipcode(),
        "insurance":   random.choice(["Medicare", "Medicaid", "Private", "Uninsured"]),
    }
    patients.append(patient)

    # ── Each patient gets 1–5 encounters (hospital visits) ───────
    num_encounters = random.randint(1, 5)

    for j in range(num_encounters):

        encounter_id   = str(uuid.uuid4())
        admit_date     = fake.date_between(start_date="-2y", end_date="today")
        length_of_stay = random.randint(1, 14)
        discharge_date = admit_date + timedelta(days=length_of_stay)

        encounter = {
            "encounter_id":    encounter_id,
            "patient_id":      patient_id,
            "encounter_type":  random.choice(ENCOUNTER_TYPES),
            "admit_date":      admit_date,
            "discharge_date":  discharge_date,
            "length_of_stay":  length_of_stay,
            "department":      random.choice(["Cardiology", "General Medicine",
                                              "Nephrology", "Pulmonology", "ICU"]),
            "discharge_status": random.choice(["Home", "Rehab Facility",
                                               "Another Hospital", "Deceased"]),
        }
        encounters.append(encounter)

        # ── Each encounter gets 2–5 lab observations ─────────────
        num_obs = random.randint(2, 5)
        sampled_labs = random.sample(LAB_TESTS, num_obs)

        for lab_name, low, high, unit in sampled_labs:
            observations.append({
                "obs_id":       str(uuid.uuid4()),
                "patient_id":   patient_id,
                "encounter_id": encounter_id,
                "test_name":    lab_name,
                "value":        round(random.uniform(low, high), 2),
                "unit":         unit,
                "obs_date":     admit_date,
            })

    # ── Each patient gets 1–4 diagnoses ──────────────────────────
    num_conditions = random.randint(1, 4)
    patient_diagnoses = random.sample(DIAGNOSES, num_conditions)

    for diagnosis in patient_diagnoses:
        conditions.append({
            "condition_id": str(uuid.uuid4()),
            "patient_id":   patient_id,
            "diagnosis":    diagnosis,
            "onset_date":   fake.date_between(start_date="-5y", end_date="-1m"),
            "is_chronic":   diagnosis in [
                "Type 2 Diabetes Mellitus", "Hypertension",
                "Coronary Artery Disease",  "Chronic Kidney Disease",
                "Heart Failure", "COPD", "Atrial Fibrillation",
            ],
        })

    # ── Each patient gets 1–3 medications ────────────────────────
    num_meds = random.randint(1, 3)
    for med in random.sample(MEDICATIONS, num_meds):
        medications.append({
            "med_id":       str(uuid.uuid4()),
            "patient_id":   patient_id,
            "medication":   med,
            "start_date":   fake.date_between(start_date="-2y", end_date="-1m"),
            "is_active":    random.choice([True, False]),
        })

# ── Save everything to CSV ────────────────────────────────────────

print("Saving CSV files...")

pd.DataFrame(patients).to_csv("data/patients.csv",         index=False)
pd.DataFrame(encounters).to_csv("data/encounters.csv",     index=False)
pd.DataFrame(conditions).to_csv("data/conditions.csv",     index=False)
pd.DataFrame(observations).to_csv("data/observations.csv", index=False)
pd.DataFrame(medications).to_csv("data/medications.csv",   index=False)

print("Done! Files saved in the data/ folder.")
print(f"  Patients:     {len(patients)}")
print(f"  Encounters:   {len(encounters)}")
print(f"  Conditions:   {len(conditions)}")
print(f"  Observations: {len(observations)}")
print(f"  Medications:  {len(medications)}")