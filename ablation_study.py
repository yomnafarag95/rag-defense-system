"""
ablation_study.py — Table VIII: Ablation study for RAG-Shield
Tests: L1-only, L2-only, L3-only, L1+L2, L1+L2+L3 (no meta), Full pipeline
"""

import json
import os
import numpy as np
from collections import defaultdict

RESULTS_FILE = "logs/eval_results.jsonl"

def load_records():
    records = []
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def compute_metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    adr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * adr * prec / (adr + prec) if (adr + prec) > 0 else 0

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "ADR": adr, "FPR": fpr, "Precision": prec, "F1": f1
    }

def simulate_layer_decisions(records, use_l1=False, use_l2=False, use_l3=False,
                              l1_thresh=0.65, l2_thresh=0.70, l3_boundary_thresh=2,
                              use_meta=False, meta_thresh=0.45):
    y_true = []
    y_pred = []

    for r in records:
        true_label = r["true_label"]
        l1 = r.get("l1_score", 0)
        l2 = r.get("l2_score", 0)
        l3 = r.get("l3_score", 0)
        risk = r.get("risk_score", 0)

        blocked = False

        if use_meta:
            # Full pipeline: use the actual risk_score from meta-aggregator
            if risk >= meta_thresh:
                blocked = True
        else:
            # Individual layer simulation
            if use_l1 and l1 >= l1_thresh:
                blocked = True
            if use_l2 and l2 >= l2_thresh:
                blocked = True
            if use_l3 and l3 >= 0.90:  # L3 high = suspicious
                blocked = True

        y_true.append(true_label)
        y_pred.append(1 if blocked else 0)

    return y_true, y_pred

def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"ERROR: {RESULTS_FILE} not found!")
        return

    records = load_records()
    print(f"Loaded {len(records)} records\n")

    # Define ablation configurations
    configs = [
        ("L1 only",           dict(use_l1=True,  use_l2=False, use_l3=False, use_meta=False)),
        ("L2 only",           dict(use_l1=False, use_l2=True,  use_l3=False, use_meta=False)),
        ("L3 only",           dict(use_l1=False, use_l2=False, use_l3=True,  use_meta=False)),
        ("L1 + L2",           dict(use_l1=True,  use_l2=True,  use_l3=False, use_meta=False)),
        ("L1 + L2 + L3",      dict(use_l1=True,  use_l2=True,  use_l3=True,  use_meta=False)),
        ("Full (meta-agg)",   dict(use_l1=False, use_l2=False, use_l3=False, use_meta=True)),
    ]

    print("=" * 78)
    print("  TABLE VIII: Ablation Study")
    print("=" * 78)
    print(f"  {'Configuration':<20} {'ADR':>7} {'FPR':>7} {'Prec':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 78)

    results = []
    for name, kwargs in configs:
        y_true, y_pred = simulate_layer_decisions(records, **kwargs)
        m = compute_metrics(y_true, y_pred)
        print(f"  {name:<20} {m['ADR']:>7.3f} {m['FPR']:>7.3f} {m['Precision']:>7.3f} {m['F1']:>7.3f} {m['TP']:>5} {m['FP']:>5} {m['FN']:>5}")
        results.append({"name": name, **m})

    print("-" * 78)

    # Analysis
    print("\n  Key Observations:")

    l1_only = results[0]
    l2_only = results[1]
    l1_l2 = results[3]
    full = results[5]

    print(f"    - L1 alone: ADR={l1_only['ADR']:.3f}, FPR={l1_only['FPR']:.3f}")
    print(f"    - L2 alone: ADR={l2_only['ADR']:.3f}, FPR={l2_only['FPR']:.3f}")
    print(f"    - L1+L2 union: ADR={l1_l2['ADR']:.3f}, FPR={l1_l2['FPR']:.3f}")
    print(f"    - Full pipeline: ADR={full['ADR']:.3f}, FPR={full['FPR']:.3f}")

    if l1_l2['ADR'] > l1_only['ADR'] and l1_l2['ADR'] > l2_only['ADR']:
        print(f"    → L1+L2 ensemble gain: +{l1_l2['ADR'] - max(l1_only['ADR'], l2_only['ADR']):.3f} ADR over best single layer")

    if full['FPR'] < l1_l2['FPR']:
        print(f"    → Meta-aggregator reduces FPR: {l1_l2['FPR']:.3f} → {full['FPR']:.3f}")

    # LaTeX output
    print("\n  --- LaTeX for TABLE VIII ---")
    print(r"  \begin{table}[h]")
    print(r"  \centering")
    print(r"  \caption{Ablation Study}")
    print(r"  \begin{tabular}{lrrrr}")
    print(r"  \hline")
    print(r"  \textbf{Configuration} & \textbf{ADR} & \textbf{FPR} & \textbf{Prec.} & \textbf{F1} \\")
    print(r"  \hline")
    for r in results:
        print(f"  {r['name']} & {r['ADR']:.3f} & {r['FPR']:.3f} & {r['Precision']:.3f} & {r['F1']:.3f} \\\\")
    print(r"  \hline")
    print(r"  \end{tabular}")
    print(r"  \end{table}")

    # Save
    with open("logs/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Saved → logs/ablation_results.json")

if __name__ == "__main__":
    main()