import os
import sys

print("=== HealthGuard Startup ===")

os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)

# Step 1 — Generate data
if not os.path.exists("data/patients.csv"):
    print("Generating patient data...")
    import subprocess
    subprocess.run([sys.executable, "scripts/generate_data.py"], check=True)
else:
    print("Data already exists, skipping.")

# Step 2 — Load database
if not os.path.exists("data/health.db"):
    print("Loading database...")
    import subprocess
    subprocess.run([sys.executable, "scripts/load_database.py"], check=True)
else:
    print("Database already exists, skipping.")

# Step 3 — Feature engineering
if not os.path.exists("data/features.csv"):
    print("Engineering features...")
    import subprocess
    subprocess.run([sys.executable, "scripts/feature_engineering.py"], check=True)
else:
    print("Features already exist, skipping.")

# Step 4 — Train model
if not os.path.exists("models/readmission_model.json"):
    print("Training model...")
    import subprocess
    subprocess.run([sys.executable, "scripts/train_model.py"], check=True)
else:
    print("Model already exists, skipping.")

print("=== Startup complete ===")