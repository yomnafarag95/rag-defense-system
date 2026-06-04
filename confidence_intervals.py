"""
confidence_intervals.py — Bootstrap 95% CIs for all metrics
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=10000, ci=0.95):
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        try:
            s = metric_fn(y_true[idx], y_pred[idx])
            scores.append(s)
        except:
            continue
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    return np.mean(scores), lower, upper

# ---------- YOUR RESULTS ----------
# Standard benchmark: 131 TP, 69 FN, 0 FP, 0 TN (attack-only set)
# Full: 131 TP, 69 FN, 0 FP, 423 TN


y_true = np.array([1]*200 + [0]*423)
y_pred = np.array([1]*142 + [0]*58 + [0]*423)

print("=" * 60)
print("  Bootstrap 95% Confidence Intervals (n_boot=10,000)")
print("=" * 60)

for name, fn in [
    ("Recall (ADR)", recall_score),
    ("Precision", precision_score),
    ("F1", f1_score),
]:
    mean, lo, hi = bootstrap_ci(y_true, y_pred, fn)
    print(f"  {name:<20}: {mean:.4f}  [{lo:.4f}, {hi:.4f}]")

# FPR
def fpr_fn(yt, yp):
    fp = ((yp == 1) & (yt == 0)).sum()
    tn = ((yp == 0) & (yt == 0)).sum()
    return fp / (fp + tn + 1e-9)

mean, lo, hi = bootstrap_ci(y_true, y_pred, fpr_fn)
print(f"  {'FPR':<20}: {mean:.4f}  [{lo:.4f}, {hi:.4f}]")

print("\n  Use these in the paper, e.g.:")
print("  'ADR = 0.655 (95% CI: [0.59, 0.72])'")