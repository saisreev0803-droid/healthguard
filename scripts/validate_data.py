import pandas as pd

df = pd.read_csv("data/features.csv")

print("Running data validation checks...")
errors = []

# Check 1 — No missing values anywhere
missing = df.isnull().sum().sum()
if missing > 0:
    errors.append(f"FAIL: {missing} missing values found")
else:
    print("  PASS: No missing values")

# Check 2 — Age must be between 18 and 100
bad_ages = df[(df["age"] < 18) | (df["age"] > 100)]
if len(bad_ages) > 0:
    errors.append(f"FAIL: {len(bad_ages)} patients with invalid age")
else:
    print("  PASS: All ages valid (18–100)")

# Check 3 — Target label must only be 0 or 1
bad_labels = df[~df["readmitted_30days"].isin([0, 1])]
if len(bad_labels) > 0:
    errors.append(f"FAIL: {len(bad_labels)} invalid target labels")
else:
    print("  PASS: Target labels are all 0 or 1")

# Check 4 — Counts must never be negative
for col in ["num_admissions_12m", "num_chronic_conditions", "num_medications"]:
    bad = df[df[col] < 0]
    if len(bad) > 0:
        errors.append(f"FAIL: Negative values in {col}")
    else:
        print(f"  PASS: {col} has no negative values")

# Check 5 — Must have exactly 1000 patients
if len(df) != 1000:
    errors.append(f"FAIL: Expected 1000 patients, got {len(df)}")
else:
    print("  PASS: Exactly 1000 patients present")

# ── Final result ──────────────────────────────────────────────────
print()
if errors:
    print("Validation FAILED:")
    for e in errors:
        print(f"  {e}")
else:
    print("All checks passed. Data is clean and ready for modeling.")