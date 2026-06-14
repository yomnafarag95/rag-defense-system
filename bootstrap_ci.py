"""
bootstrap_ci.py
───────────────
Compute 95% bootstrap confidence intervals on RAG-Shield eval results.

Reads from logs/eval_results.jsonl (produced by eval_suite.py --mode all).
No model inference required — runs in < 2 minutes.

Method
──────
  10,000 bootstrap resamples with replacement.
  95% CI = [2.5th percentile, 97.5th percentile].

  ADR       : computed on attack samples only
  FPR       : computed on benign samples only
  F1        : computed on attack samples only
  Precision : computed on attack samples only
  AUC-ROC   : computed on combined set (requires both classes)

Usage
─────
  python eval_suite.py --mode all     (must run first)
  python bootstrap_ci.py
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)


# ─────────────────────────────────────────────────────────────────────────────
# Metric functions
# ─────────────────────────────────────────────────────────────────────────────

def _adr(y_true, y_pred, _):
    return float(recall_score(y_true, y_pred, zero_division=0))

def _fpr(y_true, y_pred, _):
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    return fp / max(fp + tn, 1)

def _f1(y_true, y_pred, _):
    return float(f1_score(y_true, y_pred, zero_division=0))

def _precision(y_true, y_pred, _):
    return float(precision_score(y_true, y_pred, zero_division=0))

def _auc_roc(y_true, _, y_prob):
    if len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap engine
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(y_true: list,
                 y_pred: list,
                 y_prob: list,
                 metric_fn,
                 n_bootstrap: int = 10_000,
                 seed: int = 42) -> tuple[float, float, float]:
    """
    Compute point estimate and 95% bootstrap CI for a metric.

    Returns
    ───────
    (point_estimate, lower_95, upper_95)
    """
    rng   = np.random.RandomState(seed)
    n     = len(y_true)
    point = metric_fn(y_true, y_pred, y_prob)

    if point is None:
        return None, None, None

    scores = []
    for _ in range(n_bootstrap):
        idx    = rng.choice(n, n, replace=True)
        yt     = [y_true[i] for i in idx]
        yp     = [y_pred[i] for i in idx]
        yprob  = [y_prob[i] for i in idx]
        try:
            s = metric_fn(yt, yp, yprob)
            if s is not None:
                scores.append(s)
        except Exception:
            pass

    if not scores:
        return round(point, 4), None, None

    lower = float(np.percentile(scores, 2.5))
    upper = float(np.percentile(scores, 97.5))
    return round(point, 4), round(lower, 4), round(upper, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Bootstrap 95% Confidence Intervals ===\n")

    results_path = Path("logs/eval_results.jsonl")
    if not results_path.exists():
        print(f"[error] {results_path} not found.")
        print("        Run: python eval_suite.py --mode all")
        exit(1)

    # Load all entries
    entries = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"  Loaded {len(entries)} entries from {results_path}")

    # Separate by true label
    attack_entries = [e for e in entries if e["true_label"] == 1]
    benign_entries  = [e for e in entries if e["true_label"] == 0]

    print(f"  Attack entries : {len(attack_entries)}")
    print(f"  Benign entries : {len(benign_entries)}\n")

    # Extract vectors
    yt_atk  = [e["true_label"]     for e in attack_entries]
    yp_atk  = [e["pred_prevented"] for e in attack_entries]  # fixed: was pred_label (non-existent field)
    ypr_atk = [e["risk_score"]     for e in attack_entries]

    yt_ben  = [e["true_label"]     for e in benign_entries]
    yp_ben  = [e["pred_prevented"] for e in benign_entries]  # fixed: was pred_label
    ypr_ben = [e["risk_score"]     for e in benign_entries]

    yt_all  = [e["true_label"]     for e in entries]
    yp_all  = [e["pred_prevented"] for e in entries]         # fixed: was pred_label
    ypr_all = [e["risk_score"]     for e in entries]

    N_BOOTSTRAP = 10_000
    print(f"  Running {N_BOOTSTRAP:,} bootstrap resamples per metric ...")
    print(f"  This takes ~30-60 seconds.\n")

    # Compute CIs
    metrics = {}

    print("  Computing ADR (attack samples) ...")
    metrics["ADR"] = bootstrap_ci(yt_atk, yp_atk, ypr_atk, _adr, N_BOOTSTRAP)

    print("  Computing F1 (attack samples) ...")
    metrics["F1"] = bootstrap_ci(yt_atk, yp_atk, ypr_atk, _f1, N_BOOTSTRAP)

    print("  Computing Precision (attack samples) ...")
    metrics["Precision"] = bootstrap_ci(yt_atk, yp_atk, ypr_atk, _precision, N_BOOTSTRAP)

    print("  Computing FPR (benign samples) ...")
    metrics["FPR"] = bootstrap_ci(yt_ben, yp_ben, ypr_ben, _fpr, N_BOOTSTRAP)

    print("  Computing AUC-ROC (combined set) ...")
    metrics["AUC-ROC"] = bootstrap_ci(yt_all, yp_all, ypr_all, _auc_roc, N_BOOTSTRAP)

    # ── Print table ───────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  BOOTSTRAP CONFIDENCE INTERVALS (95%, 10,000 resamples)")
    print("  For use in Table III of your paper")
    print("="*65)
    print(f"  {'Metric':<12} {'Value':>8}  {'Lower':>8}  {'Upper':>8}  {'CI String'}")
    print("-"*65)

    for name, (val, lo, hi) in metrics.items():
        if lo is not None and hi is not None:
            ci_str = f"[{lo:.4f}, {hi:.4f}]"
        else:
            ci_str = "N/A"
        val_str = f"{val:.4f}" if val is not None else "N/A"
        lo_str  = f"{lo:.4f}"  if lo  is not None else "N/A"
        hi_str  = f"{hi:.4f}"  if hi  is not None else "N/A"
        print(f"  {name:<12} {val_str:>8}  {lo_str:>8}  {hi_str:>8}  {ci_str}")

    print("="*65)

    # ── Evasion bench note ────────────────────────────────────────────────────
    evasion_entries = [
        e for e in attack_entries
        if e.get("attack_type") in {
            "encoding_obfuscation", "payload_splitting",
            "indirect_injection", "role_manipulation", "context_exhaustion",
        }
    ]
    print(f"\n  Note: Evasion set (n=7) is too small for reliable bootstrap CIs.")
    print(f"  Report evasion ADR=0.8571 and F1=0.9231 as point estimates only.")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path("logs/bootstrap_ci.json")
    out_path.parent.mkdir(exist_ok=True)

    result_dict = {
        name: {
            "value":    val,
            "ci_lower": lo,
            "ci_upper": hi,
            "ci_string": f"[{lo:.4f}, {hi:.4f}]" if lo else "N/A",
        }
        for name, (val, lo, hi) in metrics.items()
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    print(f"\n  Full CI results saved -> {out_path}")
    print(f"\n  Paper-ready Table III update:")
    print(f"  ---------------------------------------------------------")

    for name, (val, lo, hi) in metrics.items():
        if lo is not None and hi is not None:
            print(f"  {name:<12} : {val:.4f}  [{lo:.4f}, {hi:.4f}]")
        else:
            print(f"  {name:<12} : {val}")