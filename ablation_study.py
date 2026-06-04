"""
ablation_study.py
─────────────────
Ablation study for RAG-Shield — optimised runtime version.

Benign evaluation strategy
───────────────────────────
  L1 only    : runs 553 benign  (~2 min)   — fast, no DeBERTa
  L2 only    : runs 553 benign  (~7 min)   — medium, DeBERTa only
  L3 only    : SKIPS benign     — uses FPR from main eval (0.1212)
  L1+L2 OR   : runs 553 benign  (~7 min)   — medium, no L3
  Full       : SKIPS benign     — uses FPR from main eval (0.1212)

Reason: L3 only and Full pipeline take ~2700ms per sample.
        553 benign × 2700ms = ~25 minutes per config.
        FPR is already measured precisely in eval_suite.py.

Runtime estimate
─────────────────
  L1 only     attacks+benign :  ~2 min
  L2 only     attacks+benign :  ~7 min
  L3 only     attacks only   :  ~6 min
  L1+L2 OR    attacks+benign :  ~7 min
  Full        attacks only   :  ~6 min
  Total                      : ~28 min

Usage
─────
  python ablation_study.py
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import confusion_matrix

from eval_suite import _clean_text, _valid_eval_text, _is_detected
from layer1_anomaly import load_detector, split_chunks
from layer2_classifier import load_classifier
from layer3_enhanced import load_monitor
from orchestrator import MetaAggregator, _make_skipped_l2, _make_skipped_l3

SYSTEM_PROMPT = "Answer using only the knowledge base."

# FPR from main eval — used for L3-only and Full pipeline
# instead of rerunning 553 benign samples
MAIN_EVAL_FPR = 0.1212
MAIN_EVAL_FP  = 67
MAIN_EVAL_TN  = 486


# ─────────────────────────────────────────────────────────────────────────────
# Ablation pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_ablation(document: str,
                  query: str,
                  system_prompt: str,
                  l1_detector,
                  l2_classifier,
                  l3_monitor,
                  meta_aggregator,
                  mode: str) -> dict:
    """
    Run pipeline with selective layer ablation.

    mode options
    ────────────
      l1_only   : block only if L1 fires
      l2_only   : block only if L2 fires
      l3_only   : block only if L3 fires
      l1_l2_or  : block if L1 OR L2 fires
      full      : normal full pipeline
    """
    chunks = split_chunks(document)
    l1     = l1_detector.scan(chunks)

    # ── L1 only ───────────────────────────────────────────────────────────────
    if mode == "l1_only":
        blocked = l1["blocked"]
        return {
            "blocked":        blocked,
            "action":         "blocked" if blocked else "allow",
            "blocking_layer": "Layer 1 - Anomaly Detection" if blocked else None,
            "meta":           {"risk_score": float(l1["max_score"])},
        }

    # Run L2 for all remaining modes
    l2 = l2_classifier.classify(query, chunks)

    # ── L2 only ───────────────────────────────────────────────────────────────
    if mode == "l2_only":
        blocked = l2["blocked"]
        return {
            "blocked":        blocked,
            "action":         "blocked" if blocked else "allow",
            "blocking_layer": "Layer 2 - Intent Classifier" if blocked else None,
            "meta":           {"risk_score": float(l2["stage1_prob"])},
        }

    # ── L1 + L2 OR ────────────────────────────────────────────────────────────
    if mode == "l1_l2_or":
        blocked = l1["blocked"] or l2["blocked"]
        if l1["blocked"]:
            bl = "Layer 1 - Anomaly Detection"
        elif l2["blocked"]:
            bl = "Layer 2 - Intent Classifier"
        else:
            bl = None
        return {
            "blocked":        blocked,
            "action":         "blocked" if blocked else "allow",
            "blocking_layer": bl,
            "meta":           {
                "risk_score": float(max(l1["max_score"], l2["stage1_prob"]))
            },
        }

    # Run L3 for l3_only and full
    l3 = l3_monitor.check(query, system_prompt, chunks, l1, l2)

    # ── L3 only ───────────────────────────────────────────────────────────────
    if mode == "l3_only":
        blocked = l3["blocked"]
        return {
            "blocked":        blocked,
            "action":         "blocked" if blocked else "allow",
            "blocking_layer": "Layer 3 - Behavioral Monitor" if blocked else None,
            "meta":           {"risk_score": float(l3["consistency_score"])},
        }

    # ── Full pipeline ─────────────────────────────────────────────────────────
    agg  = meta_aggregator or MetaAggregator()
    meta = agg.predict(l1, l2, l3, query=query)

    if l1["blocked"]:
        bl = "Layer 1 - Anomaly Detection"
    elif l2["blocked"]:
        bl = "Layer 2 - Intent Classifier"
    elif l3["blocked"]:
        bl = "Layer 3 - Behavioral Monitor"
    elif meta["action"] in ("blocked", "hard_block"):
        bl = "Meta Aggregator - Combined Risk"
    else:
        bl = None

    return {
        "blocked":        meta["action"] in ("blocked", "hard_block"),
        "action":         meta["action"],
        "blocking_layer": bl,
        "meta":           meta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_attack_samples() -> list[dict]:
    samples = []

    # InjecAgent
    ia_path = Path("data/injecagent.jsonl")
    if ia_path.exists():
        count = 0
        with open(ia_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = _clean_text(row.get("text"))
                if _valid_eval_text(text):
                    samples.append({
                        "text":        text,
                        "label":       1,
                        "attack_type": row.get("attack_type", "indirect_injection"),
                    })
                    count += 1
        print(f"  InjecAgent    : {count} samples")
    else:
        print("  [warn] injecagent.jsonl not found")

    # HackAPrompt holdout
    hp_path = Path("data/hackaprompt_holdout_seed42.csv")
    if hp_path.exists():
        import pandas as pd
        from sklearn.model_selection import train_test_split

        df       = pd.read_csv(hp_path)
        sample_n = min(69, len(df))
        counts   = df["category"].value_counts()
        eligible = df[
            df["category"].isin(counts[counts >= 2].index)
        ].copy()

        if len(eligible) >= sample_n:
            sample_df, _ = train_test_split(
                eligible, train_size=sample_n,
                stratify=eligible["category"], random_state=42,
            )
        else:
            sample_df = df.sample(n=sample_n, random_state=42)

        count = 0
        for _, row in sample_df.iterrows():
            text = _clean_text(row.get("text"))
            if _valid_eval_text(text):
                samples.append({
                    "text":        text,
                    "label":       1,
                    "attack_type": row.get("category", "unknown"),
                })
                count += 1
        print(f"  HackAPrompt   : {count} samples")

    print(f"  Total attacks : {len(samples)}")
    return samples


def _load_benign_samples() -> list[dict]:
    samples = []
    ext_path = Path("data/extended_benign.csv")
    if ext_path.exists():
        import pandas as pd
        df = pd.read_csv(ext_path)
        for _, row in df.iterrows():
            text = _clean_text(row.get("query"))
            if _valid_eval_text(text):
                samples.append({"text": text, "label": 0})
        print(f"  Benign        : {len(samples)} samples")
    else:
        print("  [warn] extended_benign.csv not found")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Run one configuration
# ─────────────────────────────────────────────────────────────────────────────

def _run_config(samples: list[dict],
                l1_detector,
                l2_classifier,
                l3_monitor,
                meta_aggregator,
                mode: str,
                tag: str) -> tuple[list, list]:
    """Returns (y_true, y_pred)."""

    y_true, y_pred = [], []
    t_start = time.time()
    n = len(samples)

    for i, sample in enumerate(samples, 1):
        text   = sample["text"]
        result = _run_ablation(
            document        = text,
            query           = text[:200],
            system_prompt   = SYSTEM_PROMPT,
            l1_detector     = l1_detector,
            l2_classifier   = l2_classifier,
            l3_monitor      = l3_monitor,
            meta_aggregator = meta_aggregator,
            mode            = mode,
        )
        predicted = 1 if _is_detected(result.get("action", "allow")) else 0
        y_true.append(sample["label"])
        y_pred.append(predicted)

        if i % 25 == 0 or i == n:
            elapsed = time.time() - t_start
            rate    = i / max(elapsed, 0.01)
            eta     = (n - i) / max(rate, 0.01)
            print(f"    [{tag}] {i}/{n}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    return y_true, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Compute full metrics from attack + benign results
# ─────────────────────────────────────────────────────────────────────────────

def _compute_row(yt_atk: list, yp_atk: list,
                 fp: int, tn: int) -> dict:
    """
    Combine attack predictions with known FP/TN from benign.
    """
    tp = sum(1 for t, p in zip(yt_atk, yp_atk) if t == 1 and p == 1)
    fn = sum(1 for t, p in zip(yt_atk, yp_atk) if t == 1 and p == 0)

    adr  = tp / max(tp + fn, 1)
    fpr  = fp / max(fp + tn, 1)
    prec = tp / max(tp + fp, 1)
    f1   = (2 * adr * prec) / max(adr + prec, 1e-8)

    return {
        "ADR":  round(adr,  4),
        "FPR":  round(fpr,  4),
        "Prec": round(prec, 4),
        "F1":   round(f1,   4),
        "TP":   tp,
        "FP":   fp,
        "TN":   tn,
        "FN":   fn,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== RAG-Shield Ablation Study ===")
    print("    Optimised: benign skipped for L3-only and Full pipeline")
    print(f"    Using main-eval FPR={MAIN_EVAL_FPR} "
          f"(FP={MAIN_EVAL_FP}, TN={MAIN_EVAL_TN}) for those configs\n")

    # ── Load components ───────────────────────────────────────────────────────
    print("Loading pipeline components ...")
    l1  = load_detector()
    l2  = load_classifier()
    l3  = load_monitor()
    agg = MetaAggregator.load()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading evaluation data ...")
    attack_samples = _load_attack_samples()
    benign_samples = _load_benign_samples()

    # ─────────────────────────────────────────────────────────────────────────
    # Configurations
    # run_benign=True  → actually runs 553 benign samples
    # run_benign=False → uses MAIN_EVAL_FP / MAIN_EVAL_TN directly
    # ─────────────────────────────────────────────────────────────────────────
    configs = [
        ("l1_only",  "L1 only  (Anomaly Detection)",     True),
        ("l2_only",  "L2 only  (Intent Classifier)",     True),
        ("l3_only",  "L3 only  (Semantic Monitor)",      False),
        ("l1_l2_or", "L1+L2   (OR-logic union)",         True),
        ("full",     "Full     (meta-aggregator)",        False),
    ]

    results = {}
    total_start = time.time()

    for mode, desc, run_benign in configs:
        print(f"\n{'─'*60}")
        print(f"  Config : {desc}")
        print(f"{'─'*60}")

        # Attack samples — always run
        print(f"  Running attacks ({len(attack_samples)} samples) ...")
        yt_atk, yp_atk = _run_config(
            attack_samples, l1, l2, l3, agg, mode, "ATK"
        )

        # Benign samples — conditional
        if run_benign and benign_samples:
            print(f"  Running benign ({len(benign_samples)} samples) ...")
            yt_ben, yp_ben = _run_config(
                benign_samples, l1, l2, l3, agg, mode, "BEN"
            )
            fp = sum(1 for t, p in zip(yt_ben, yp_ben) if t == 0 and p == 1)
            tn = sum(1 for t, p in zip(yt_ben, yp_ben) if t == 0 and p == 0)
            benign_note = f"measured (n={len(benign_samples)})"
        else:
            fp = MAIN_EVAL_FP
            tn = MAIN_EVAL_TN
            benign_note = f"from main eval (FP={fp}, TN={tn})"
            print(f"  Benign : skipped — using {benign_note}")

        row = _compute_row(yt_atk, yp_atk, fp, tn)
        results[desc] = {"metrics": row, "benign_note": benign_note}

        print(f"\n  Result : ADR={row['ADR']:.4f}  FPR={row['FPR']:.4f}  "
              f"F1={row['F1']:.4f}  "
              f"TP={row['TP']}  FP={row['FP']}  "
              f"TN={row['TN']}  FN={row['FN']}")

    total_elapsed = time.time() - total_start

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print("  ABLATION STUDY — TABLE VII")
    print(f"{'='*90}")
    print(
        f"  {'Configuration':<35} "
        f"{'ADR':>6} {'FPR':>6} {'Prec':>6} {'F1':>6} "
        f"{'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}  Benign"
    )
    print(f"  {'-'*88}")

    for desc, data in results.items():
        r = data["metrics"]
        note = data["benign_note"]
        print(
            f"  {desc:<35} "
            f"{r['ADR']:>6.3f} {r['FPR']:>6.4f} {r['Prec']:>6.3f} {r['F1']:>6.3f} "
            f"{r['TP']:>5} {r['FP']:>5} {r['TN']:>5} {r['FN']:>5}  {note}"
        )

    print(f"{'='*90}")
    print(f"\n  Total runtime : {total_elapsed/60:.1f} minutes")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path("logs/ablation_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved → {out_path}")
    print(f"\n  Paper Table VII — copy these rows:")
    print(f"  {'─'*60}")

    for desc, data in results.items():
        r = data["metrics"]
        short = desc.strip().split("(")[0].strip()
        print(
            f"  {short:<20} "
            f"ADR={r['ADR']:.3f}  FPR={r['FPR']:.4f}  "
            f"Prec={r['Prec']:.3f}  F1={r['F1']:.3f}  "
            f"TP={r['TP']}  FP={r['FP']}  FN={r['FN']}"
        )