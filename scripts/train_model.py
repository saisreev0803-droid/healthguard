import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────────
# STEP A — Load the feature table
# ─────────────────────────────────────────────────────────────────

print("Loading features...")
df = pd.read_csv("data/features.csv")

# Drop non-numeric columns the model cannot use
# patient_id is just an identifier, gender is already encoded
df = df.drop(columns=["patient_id", "gender"])

# Separate features (X) from the target label (y)
# X = everything the model uses as input
# y = what the model is trying to predict
X = df.drop(columns=["readmitted_30days"])
y = df["readmitted_30days"]

print(f"  Features shape: {X.shape}")
print(f"  Readmission rate: {y.mean() * 100:.1f}%")
print(f"  Feature names: {list(X.columns)}")

# ─────────────────────────────────────────────────────────────────
# STEP B — Split into training and test sets
# ─────────────────────────────────────────────────────────────────
# test_size=0.2 means 20% goes to test, 80% to training
# stratify=y means the readmission ratio stays the same in both splits
# random_state=42 makes the split reproducible (same split every run)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"\nTrain size: {len(X_train)} patients")
print(f"Test size:  {len(X_test)} patients")

# ─────────────────────────────────────────────────────────────────
# STEP C — Handle class imbalance
# ─────────────────────────────────────────────────────────────────
# About 80% of patients are NOT readmitted (label 0)
# Only 20% ARE readmitted (label 1)
# Without correction the model learns to just predict 0 every time
# scale_pos_weight tells XGBoost to pay more attention to the minority class

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()
scale = negative_count / positive_count

print(f"\nClass balance — Not readmitted: {negative_count}, Readmitted: {positive_count}")
print(f"scale_pos_weight: {scale:.2f}")

# ─────────────────────────────────────────────────────────────────
# STEP D — Define the model with hyperparameters
# ─────────────────────────────────────────────────────────────────
# These are the settings that control how the model learns.
# Think of them as dials you can tune to improve performance.

model = xgb.XGBClassifier(
    n_estimators=200,        # number of trees in the chain
    max_depth=4,             # how deep each tree can grow (deeper = more complex)
    learning_rate=0.05,      # how much each tree corrects the previous one
    subsample=0.8,           # use 80% of rows when building each tree (prevents overfitting)
    colsample_bytree=0.8,    # use 80% of features per tree (prevents overfitting)
    scale_pos_weight=scale,  # handles class imbalance
    use_label_encoder=False,
    eval_metric="auc",
    random_state=42,
)

# ─────────────────────────────────────────────────────────────────
# STEP E — Cross-validation before final training
# ─────────────────────────────────────────────────────────────────
# Instead of training once and hoping for the best, we train 5 times
# on different slices of the training data. This gives a reliable
# estimate of true performance before we ever touch the test set.

print("\nRunning 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

print(f"  CV AUC scores: {cv_scores.round(3)}")
print(f"  Mean AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# ─────────────────────────────────────────────────────────────────
# STEP F — Train the final model and log everything with MLflow
# ─────────────────────────────────────────────────────────────────

print("\nTraining final model...")

os.makedirs("models", exist_ok=True)

with mlflow.start_run(run_name="xgboost_readmission_v1"):

    # Log the hyperparameters
    mlflow.log_param("n_estimators",     200)
    mlflow.log_param("max_depth",        4)
    mlflow.log_param("learning_rate",    0.05)
    mlflow.log_param("subsample",        0.8)
    mlflow.log_param("colsample_bytree", 0.8)

    # Train on the full training set
    model.fit(X_train, y_train)

    # ── Evaluate on the test set ──────────────────────────────────
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # probability of readmission
    y_pred       = model.predict(X_test)              # hard 0/1 prediction

    auc   = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    precision = report["1"]["precision"]
    recall    = report["1"]["recall"]
    f1        = report["1"]["f1-score"]

    print(f"\n  Test AUC:       {auc:.3f}")
    print(f"  Precision:      {precision:.3f}")
    print(f"  Recall:         {recall:.3f}")
    print(f"  F1 Score:       {f1:.3f}")

    # Log the metrics
    mlflow.log_metric("test_auc",   auc)
    mlflow.log_metric("precision",  precision)
    mlflow.log_metric("recall",     recall)
    mlflow.log_metric("f1",         f1)
    mlflow.log_metric("cv_auc_mean", cv_scores.mean())

    # Save the model
    mlflow.xgboost.log_model(model, "model")
    model.save_model("models/readmission_model.json")
    print("\n  Model saved to models/readmission_model.json")

# ─────────────────────────────────────────────────────────────────
# STEP G — Confusion matrix chart
# ─────────────────────────────────────────────────────────────────
# A confusion matrix shows the 4 types of predictions:
# True Positive:  model said readmitted, patient WAS readmitted  (good)
# True Negative:  model said fine,       patient was FINE        (good)
# False Positive: model said readmitted, patient was FINE        (unnecessary intervention)
# False Negative: model said fine,       patient WAS readmitted  (missed case - dangerous)

os.makedirs("notebooks", exist_ok=True)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Predicted: No", "Predicted: Yes"],
    yticklabels=["Actual: No",    "Actual: Yes"],
)
plt.title("Confusion Matrix — 30-Day Readmission")
plt.tight_layout()
plt.savefig("notebooks/confusion_matrix.png")
plt.close()
print("  Chart saved: notebooks/confusion_matrix.png")

# ─────────────────────────────────────────────────────────────────
# STEP H — ROC curve chart
# ─────────────────────────────────────────────────────────────────
# The ROC curve shows the tradeoff between catching true positives
# and accidentally flagging healthy patients.
# AUC (Area Under Curve) summarises it as one number:
#   0.5 = random guessing, 1.0 = perfect model

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"XGBoost (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve — 30-Day Readmission Model")
plt.legend()
plt.tight_layout()
plt.savefig("notebooks/roc_curve.png")
plt.close()
print("  Chart saved: notebooks/roc_curve.png")

# ─────────────────────────────────────────────────────────────────
# STEP I — SHAP values (feature importance + explanations)
# ─────────────────────────────────────────────────────────────────

print("\nCalculating SHAP values (this takes ~30 seconds)...")

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# ── Chart 1: Global feature importance ───────────────────────────
# Shows which features matter most ACROSS ALL patients on average

plt.figure()
shap.summary_plot(
    shap_values, X_test,
    plot_type="bar",
    show=False
)
plt.title("Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig("notebooks/shap_importance.png", bbox_inches="tight")
plt.close()
print("  Chart saved: notebooks/shap_importance.png")

# ── Chart 2: SHAP dot plot ────────────────────────────────────────
# Shows direction: does a HIGH value for this feature increase or
# decrease risk? Red = high feature value, Blue = low feature value

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Summary — Feature Impact Direction")
plt.tight_layout()
plt.savefig("notebooks/shap_summary.png", bbox_inches="tight")
plt.close()
print("  Chart saved: notebooks/shap_summary.png")

# ── Chart 3: Single patient explanation ──────────────────────────
# Pick the first high-risk patient and explain exactly why
# the model gave them a high readmission probability

high_risk_indices = np.where(y_pred_proba > 0.6)[0]

if len(high_risk_indices) > 0:
    idx = high_risk_indices[0]
    plt.figure()
    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_values[idx],
            base_values   = explainer.expected_value,
            data          = X_test.iloc[idx],
            feature_names = X_test.columns.tolist()
        ),
        show=False
    )
    plt.title(f"Why is Patient {idx} High Risk?")
    plt.tight_layout()
    plt.savefig("notebooks/shap_single_patient.png", bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: notebooks/shap_single_patient.png")
    print(f"  Patient {idx} readmission probability: {y_pred_proba[idx]:.2%}")

print("\nPhase 3 complete! All charts saved in notebooks/")