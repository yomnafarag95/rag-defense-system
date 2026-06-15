"""
inspect_meta_model.py
Inspect the meta-aggregator's coefficients, intercept, and scaler parameters
to understand why it outputs risk=1.0 for all-zero feature vectors.
Run: .venv311\Scripts\python.exe inspect_meta_model.py
"""
import numpy as np
import joblib
from pathlib import Path

print("=== META-AGGREGATOR MODEL INSPECTION ===\n")

# Load model and scaler
model  = joblib.load("models/meta_aggregator.pkl")
scaler = joblib.load("models/meta_scaler.pkl") if Path("models/meta_scaler.pkl").exists() else None

print(f"Model type: {type(model)}")

# Unwrap CalibratedClassifierCV if needed
if hasattr(model, 'calibrated_classifiers_'):
    print(f"Calibrated classifiers: {len(model.calibrated_classifiers_)}")
    for i, cc in enumerate(model.calibrated_classifiers_):
        base = cc.estimator if hasattr(cc, 'estimator') else getattr(cc, 'base_estimator', None)
        if base is not None and hasattr(base, 'coef_'):
            print(f"\n  Classifier {i} coefficients: {base.coef_}")
            print(f"  Classifier {i} intercept:    {base.intercept_}")
elif hasattr(model, 'coef_'):
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept:    {model.intercept_}")

if scaler:
    print(f"\nScaler mean_:  {scaler.mean_}")
    print(f"Scaler scale_: {scaler.scale_}")
    print(f"Scaler var_:   {scaler.var_}")

# ── Test the model with known feature vectors ─────────────────────────────────
feature_names = ["l1_max","l1_win","l1_full","l2_stage1","l2_consist",
                 "l3_schema","l3_bound","l3_consist","l1xl2","l1xl3"]

test_cases = [
    ("ALL ZEROS (totally benign)           ", np.zeros(10)),
    ("All-zero except l2_consist=0.6       ", [0,0,0,0,0.6,0,0,0,0,0]),
    ("All-zero except l2_consist=1.0       ", [0,0,0,0,1.0,0,0,0,0,0]),
    ("l2_stage1=1.0 (clear attack)         ", [0,0,0,1,0.8,0,0,0,0,0]),
    ("l2_stage1=0.5 (uncertain)            ", [0,0,0,0.5,0.5,0,0,0,0,0]),
    ("All high (definite attack)           ", [0.9,0.9,0.9,0.99,0.9,1,1,0.9,0.89,0.81]),
    ("Perfect benign (query=doc, l2=0)     ", [0,0,0,0,0.05,0,0,0,0,0]),
]

print(f"\n{'='*65}")
print("  Feature vector prediction tests")
print(f"{'='*65}")
for name, fv in test_cases:
    fv = np.array(fv, dtype=float).reshape(1, -1)
    fv_scaled = scaler.transform(fv) if scaler else fv
    prob = model.predict_proba(fv_scaled)[0][1]
    print(f"  {name}  risk={prob:.4f}")

# ── Show what the scaler does to all-zeros ────────────────────────────────────
if scaler:
    zeros = np.zeros((1, 10))
    scaled_zeros = scaler.transform(zeros)
    print(f"\nScaler(all-zeros) = {scaled_zeros[0].round(4)}")
    # This reveals which direction the scaler pushes the features

print("\nDone.")
