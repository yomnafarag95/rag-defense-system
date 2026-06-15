"""
compare_baselines.py
────────────────────
Head-to-head comparison of RAG-Shield against published guardrail baselines.

Baseline systems evaluated:
  1. PromptGuard (Meta, facebook/prompt-guard-86M) — fine-tuned BERT for injection
  2. Keyword blocklist (naive baseline) — high-precision keyword list only
  3. DeBERTa standalone (our L2 component in isolation)
  4. RAG-Shield (full pipeline)

Roadmap reference: Tier 3 — Item 5 (Benchmark Comparison with Published Baselines)

Usage
─────
    python compare_baselines.py                         # all baselines
    python compare_baselines.py --skip-promptguard     # skip HF download
    python compare_baselines.py --n-attacks 50         # smaller quick run

Output
──────
    logs/baseline_comparison.json   ← full metrics per system
    Prints a formatted comparison table to stdout

Dataset
───────
    Uses the same evaluation splits as eval_suite.py:
      - Attacks  : InjecAgent (indirect) + HackAPrompt holdout (direct)
      - Benign   : data/extended_benign.csv + data/benign_queries.jsonl

Notes
─────
  - LLM-Guard is heavyweight (requires additional pip installs).
    Use --include-llmguard to enable it explicitly.
  - PromptGuard downloads ~86M params from HuggingFace on first run.
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ── Fix Windows cp1252 UnicodeEncodeError ─────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7 fallback

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading (reuse eval_suite helpers)
# ─────────────────────────────────────────────────────────────────────────────

def _load_attack_samples(n: Optional[int] = None) -> List[Dict]:
    """Load attack samples from all available holdout sources."""
    from eval_suite import _clean_text, _valid_eval_text
    samples = []

    # InjecAgent — indirect injection
    ia_path = Path("data/injecagent.jsonl")
    if ia_path.exists():
        with open(ia_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    text = _clean_text(row.get("text") or row.get("injected_prompt", ""))
                    if _valid_eval_text(text):
                        samples.append({"text": text, "label": 1, "source": "injecagent"})
                except (json.JSONDecodeError, TypeError):
                    continue

    # HackAPrompt holdout — direct injection / jailbreak
    hp_path = Path("data/hackaprompt_holdout_seed42.csv")
    if hp_path.exists():
        import pandas as pd
        df = pd.read_csv(hp_path)
        needed = (n - len(samples)) if n is not None else len(df)
        if needed > 0:
            for _, row in df.head(needed).iterrows():
                text = _clean_text(str(row.get("text", "")))
                if _valid_eval_text(text):
                    samples.append({"text": text, "label": 1, "source": "hackaprompt"})

    if n:
        samples = samples[:n]
    print(f"  Attack samples  : {len(samples)}")
    return samples


def _load_benign_samples(n: Optional[int] = None) -> List[Dict]:
    """Load benign samples."""
    from eval_suite import _clean_text, _valid_eval_text
    samples = []

    ext_path = Path("data/extended_benign.csv")
    if ext_path.exists():
        import pandas as pd
        df = pd.read_csv(ext_path)
        for _, row in df.iterrows():
            text = _clean_text(str(row.get("query", "")))
            if _valid_eval_text(text):
                samples.append({"text": text, "label": 0, "source": "extended_benign"})

    benign_path = Path("data/benign_queries.jsonl")
    if benign_path.exists():
        with open(benign_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    text = _clean_text(row.get("text") or row.get("query", ""))
                    if _valid_eval_text(text):
                        samples.append({"text": text, "label": 0, "source": "benign_queries"})
                except (json.JSONDecodeError, TypeError):
                    continue

    if n:
        samples = samples[:n]
    print(f"  Benign samples  : {len(samples)}")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(y_true: List[int], y_pred: List[int],
                     y_prob: Optional[List[float]] = None) -> Dict:
    """Compute ADR (recall), FPR, precision, F1, and AUC."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    adr  = tp / max(tp + fn, 1)     # Attack Detection Rate = Recall
    fpr  = fp / max(fp + tn, 1)     # False Positive Rate
    prec = tp / max(tp + fp, 1)     # Precision
    f1   = (2 * adr * prec) / max(adr + prec, 1e-8)
    auc  = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    return {
        "ADR":  round(adr,  4),
        "FPR":  round(fpr,  4),
        "Prec": round(prec, 4),
        "F1":   round(f1,   4),
        "AUC":  round(auc,  4) if auc is not None else None,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "n_attack": int(tp + fn),
        "n_benign": int(fp + tn),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1: Keyword blocklist
# ─────────────────────────────────────────────────────────────────────────────

def _eval_keyword(samples: List[Dict]) -> Tuple[List[int], List[float]]:
    """Evaluate the keyword-only detector (our HIGH_CONFIDENCE keywords at 0.55 threshold)."""
    from keyword_detector import keyword_check
    preds, probs = [], []
    for s in samples:
        _, _, score = keyword_check(s["text"])
        probs.append(score)
        preds.append(1 if score >= 0.55 else 0)
    return preds, probs


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2: DeBERTa standalone (L2 only)
# ─────────────────────────────────────────────────────────────────────────────

def _eval_deberta_standalone(samples: List[Dict],
                              clf) -> Tuple[List[int], List[float]]:
    """Evaluate DeBERTa Layer 2 in isolation (no L1/L3/meta)."""
    from config import L2_STAGE1_THRESHOLD
    preds, probs = [], []
    for s in samples:
        result = clf.classify(s["text"], [])
        prob = result["stage1_prob"]
        probs.append(prob)
        preds.append(1 if prob >= L2_STAGE1_THRESHOLD else 0)
    return preds, probs


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 3: PromptGuard (facebook/prompt-guard-86M)
# ─────────────────────────────────────────────────────────────────────────────

def _eval_promptguard(samples: List[Dict]) -> Tuple[List[int], List[float]]:
    """
    Evaluate Meta's PromptGuard model.
    Labels: INJECTION (attack), JAILBREAK (attack), BENIGN (safe).
    """
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        raise RuntimeError("transformers not installed.")

    print("  [PromptGuard] Loading facebook/prompt-guard-86M ...")
    try:
        pg = hf_pipeline(
            "text-classification",
            model="meta-llama/Prompt-Guard-86M",
            device=-1,
        )
    except Exception as e:
        print(f"  [PromptGuard] meta-llama/Prompt-Guard-86M failed ({e})")
        print("  [PromptGuard] Trying facebook/prompt-guard-86M ...")
        try:
            pg = hf_pipeline(
                "text-classification",
                model="facebook/prompt-guard-86M",
                device=-1,
            )
        except Exception as e2:
            raise RuntimeError(
                f"Could not load PromptGuard: {e2}\n"
                "Make sure you accepted the model's license on HuggingFace\n"
                "and run: huggingface-cli login"
            ) from e2

    ATTACK_LABELS = {"INJECTION", "JAILBREAK", "injection", "jailbreak", "LABEL_1"}
    preds, probs = [], []
    t0 = time.time()
    for i, s in enumerate(samples, 1):
        try:
            result = pg(s["text"][:512], truncation=True)[0]
            label  = result["label"].upper()
            score  = result["score"]
            is_atk = label in ATTACK_LABELS
            prob   = score if is_atk else 1.0 - score
            preds.append(1 if is_atk else 0)
            probs.append(prob)
        except Exception as exc:
            preds.append(0)
            probs.append(0.5)
        if i % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [PromptGuard] {i}/{len(samples)} ({elapsed:.0f}s elapsed)")
    return preds, probs


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 4: LLM-Guard (optional)
# ─────────────────────────────────────────────────────────────────────────────

def _eval_llmguard(samples: List[Dict]) -> Tuple[List[int], List[float]]:
    """
    Evaluate LLM-Guard's PromptInjectionScanner.
    Requires: pip install llm-guard
    """
    try:
        from llm_guard.input_scanners import PromptInjection
        from llm_guard.input_scanners.prompt_injection import MatchType
    except ImportError:
        raise RuntimeError(
            "llm-guard is not installed.\n"
            "Install with: pip install llm-guard\n"
            "Or skip with: --skip-llmguard"
        )

    print("  [LLM-Guard] Loading PromptInjection scanner ...")
    scanner = PromptInjection(threshold=0.5, match_type=MatchType.FULL)

    preds, probs = [], []
    for s in samples:
        try:
            sanitized_prompt, results_valid, results_score = scanner.scan(
                prompt=s["text"]
            )
            # results_valid=False means injection detected; score = risk
            is_attack = not results_valid
            score = results_score.get("PromptInjection", 0.5)
            preds.append(1 if is_attack else 0)
            probs.append(float(score))
        except Exception:
            preds.append(0)
            probs.append(0.5)
    return preds, probs


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 5: RAG-Shield full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _eval_ragshield_full(samples: List[Dict],
                          l1, l2, l3, meta_agg) -> Tuple[List[int], List[float]]:
    """Evaluate RAG-Shield's full 3-layer pipeline."""
    from orchestrator import run_pipeline
    from eval_suite import _is_detected

    SYSTEM_PROMPT = "Answer using only the provided knowledge base."
    BENIGN_DOC    = "This document contains standard company information."

    preds, probs = [], []
    t0 = time.time()
    for i, s in enumerate(samples, 1):
        text = s["text"]
        if s["label"] == 1:
            # Attack: text is the query; use benign doc
            doc, query = BENIGN_DOC, text
        else:
            # Benign: text is the query; use benign doc
            doc, query = BENIGN_DOC, text

        try:
            result = run_pipeline(
                document=doc,
                query=query,
                system_prompt=SYSTEM_PROMPT,
                l1_detector=l1,
                l2_classifier=l2,
                l3_monitor=l3,
                meta_aggregator=meta_agg,
            )
            prob  = result["meta"]["risk_score"]
            is_blocked = _is_detected(result.get("action", "allow"))
            preds.append(1 if is_blocked else 0)
            probs.append(prob)
        except Exception as exc:
            preds.append(0)
            probs.append(0.5)

        if i % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [RAG-Shield] {i}/{len(samples)} ({elapsed:.0f}s elapsed)")
    return preds, probs


# ─────────────────────────────────────────────────────────────────────────────
# Print comparison table
# ─────────────────────────────────────────────────────────────────────────────

def _print_table(results: Dict[str, Dict]) -> None:
    SEP = "-" * 95
    HDR = "-" * 95
    print(f"\n{'='*95}")
    print("  BASELINE COMPARISON TABLE")
    print(f"{'='*95}")
    print(
        f"  {'System':<30} {'ADR':>6} {'FPR':>6} {'Prec':>6} {'F1':>6} "
        f"{'AUC':>6} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}"
    )
    print(f"  {HDR}")

    for name, data in results.items():
        if "error" in data:
            print(f"  {name:<30} ERROR: {data['error']}")
            continue
        m = data["metrics"]
        auc_str = f"{m['AUC']:.4f}" if m["AUC"] is not None else "  N/A"
        print(
            f"  {name:<30} {m['ADR']:>6.3f} {m['FPR']:>6.4f} {m['Prec']:>6.3f} "
            f"{m['F1']:>6.3f} {auc_str:>6} "
            f"{m['TP']:>5} {m['FP']:>5} {m['TN']:>5} {m['FN']:>5}"
        )
    print(f"{'='*95}")

    # Latency table
    print(f"\n  {'System':<30} {'Mean latency':>15} {'Samples':>10}")
    print(f"  {SEP[:60]}")
    for name, data in results.items():
        if "latency_ms" in data:
            n = data.get("n_samples", "?")
            print(f"  {name:<30} {data['latency_ms']:>12.1f}ms {n:>10}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Head-to-head baseline comparison for RAG-Shield."
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000,
        help="Limit total number of samples (default: 1000).",
    )
    parser.add_argument(
        "--n-attacks", type=int, default=None,
        help="Limit number of attack samples (default: all).",
    )
    parser.add_argument(
        "--n-benign", type=int, default=None,
        help="Limit number of benign samples (default: all).",
    )
    parser.add_argument(
        "--skip-promptguard", action="store_true",
        help="Skip PromptGuard evaluation (avoids HuggingFace download).",
    )
    parser.add_argument(
        "--include-llmguard", action="store_true",
        help="Include LLM-Guard evaluation (requires: pip install llm-guard).",
    )
    parser.add_argument(
        "--skip-ragshield", action="store_true",
        help="Skip RAG-Shield full pipeline (fast mode, no model loading).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  RAG-Shield: Baseline Comparison Study")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[Data] Loading evaluation splits ...")
    n_atk = args.n_attacks if args.n_attacks is not None else (args.n_samples // 2)
    n_ben = args.n_benign if args.n_benign is not None else (args.n_samples - (args.n_samples // 2))
    attack_samples = _load_attack_samples(n_atk)
    benign_samples = _load_benign_samples(n_ben)
    all_samples    = attack_samples + benign_samples
    y_true = [s["label"] for s in all_samples]

    print(f"\n  Total samples   : {len(all_samples)} "
          f"({len(attack_samples)} attacks, {len(benign_samples)} benign)")

    # ── Load RAG-Shield components (shared across baselines 3 & 4) ────────────
    if not args.skip_ragshield:
        print("\n[Components] Loading RAG-Shield layers ...")
        from layer1_anomaly import load_detector
        from layer2_classifier import load_classifier
        from layer3_enhanced import load_monitor
        from orchestrator import MetaAggregator
        l1      = load_detector()
        l2_clf  = load_classifier()
        l3      = load_monitor()
        meta    = MetaAggregator.load()
    else:
        l1 = l2_clf = l3 = meta = None

    results = {}

    # ── Baseline 1: Keyword blocklist ──────────────────────────────────────────
    print("\n[1/5] Keyword Blocklist ...")
    t0 = time.time()
    kw_preds, kw_probs = _eval_keyword(all_samples)
    kw_ms = (time.time() - t0) / len(all_samples) * 1000
    results["Keyword Blocklist"] = {
        "metrics":    _compute_metrics(y_true, kw_preds, kw_probs),
        "latency_ms": round(kw_ms, 2),
        "n_samples":  len(all_samples),
    }
    m = results["Keyword Blocklist"]["metrics"]
    print(f"  ADR={m['ADR']:.3f}  FPR={m['FPR']:.4f}  F1={m['F1']:.3f}")

    # ── Baseline 2: DeBERTa standalone ────────────────────────────────────────
    if l2_clf is not None:
        print("\n[2/5] DeBERTa Standalone (L2 only) ...")
        t0 = time.time()
        db_preds, db_probs = _eval_deberta_standalone(all_samples, l2_clf)
        db_ms = (time.time() - t0) / len(all_samples) * 1000
        results["DeBERTa (L2 only)"] = {
            "metrics":    _compute_metrics(y_true, db_preds, db_probs),
            "latency_ms": round(db_ms, 2),
            "n_samples":  len(all_samples),
        }
        m = results["DeBERTa (L2 only)"]["metrics"]
        print(f"  ADR={m['ADR']:.3f}  FPR={m['FPR']:.4f}  F1={m['F1']:.3f}")
    else:
        print("\n[2/5] DeBERTa Standalone — skipped (--skip-ragshield).")

    # ── Baseline 3: PromptGuard ────────────────────────────────────────────────
    if not args.skip_promptguard:
        print("\n[3/5] PromptGuard (facebook/prompt-guard-86M) ...")
        try:
            t0 = time.time()
            pg_preds, pg_probs = _eval_promptguard(all_samples)
            pg_ms = (time.time() - t0) / len(all_samples) * 1000
            results["PromptGuard (86M)"] = {
                "metrics":    _compute_metrics(y_true, pg_preds, pg_probs),
                "latency_ms": round(pg_ms, 2),
                "n_samples":  len(all_samples),
            }
            m = results["PromptGuard (86M)"]["metrics"]
            print(f"  ADR={m['ADR']:.3f}  FPR={m['FPR']:.4f}  F1={m['F1']:.3f}")
        except Exception as exc:
            print(f"  [ERROR] PromptGuard failed: {exc}")
            results["PromptGuard (86M)"] = {"error": str(exc)}
    else:
        print("\n[3/5] PromptGuard — skipped (--skip-promptguard).")

    # ── Baseline 4: LLM-Guard ─────────────────────────────────────────────────
    if args.include_llmguard:
        print("\n[4/5] LLM-Guard (PromptInjection scanner) ...")
        try:
            t0 = time.time()
            lg_preds, lg_probs = _eval_llmguard(all_samples)
            lg_ms = (time.time() - t0) / len(all_samples) * 1000
            results["LLM-Guard"] = {
                "metrics":    _compute_metrics(y_true, lg_preds, lg_probs),
                "latency_ms": round(lg_ms, 2),
                "n_samples":  len(all_samples),
            }
            m = results["LLM-Guard"]["metrics"]
            print(f"  ADR={m['ADR']:.3f}  FPR={m['FPR']:.4f}  F1={m['F1']:.3f}")
        except Exception as exc:
            print(f"  [ERROR] LLM-Guard failed: {exc}")
            results["LLM-Guard"] = {"error": str(exc)}
    else:
        print("\n[4/5] LLM-Guard — skipped (use --include-llmguard to enable).")

    # ── Baseline 5: RAG-Shield full ────────────────────────────────────────────
    if not args.skip_ragshield:
        print(f"\n[5/5] RAG-Shield (full pipeline) ...")
        t0 = time.time()
        rs_preds, rs_probs = _eval_ragshield_full(all_samples, l1, l2_clf, l3, meta)
        rs_ms = (time.time() - t0) / len(all_samples) * 1000
        results["RAG-Shield (full)"] = {
            "metrics":    _compute_metrics(y_true, rs_preds, rs_probs),
            "latency_ms": round(rs_ms, 2),
            "n_samples":  len(all_samples),
        }
        m = results["RAG-Shield (full)"]["metrics"]
        print(f"  ADR={m['ADR']:.3f}  FPR={m['FPR']:.4f}  F1={m['F1']:.3f}")
    else:
        print("\n[5/5] RAG-Shield full — skipped (--skip-ragshield).")

    # ── Print & save results ───────────────────────────────────────────────────
    _print_table(results)

    out_path = Path("logs/baseline_comparison.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved -> {out_path}")

    # ── Paper-ready text ───────────────────────────────────────────────────────
    print("\n  Paper Table (copy-paste):")
    print(f"  {'-'*60}")
    for name, data in results.items():
        if "error" in data:
            continue
        m = data["metrics"]
        lat = f"{data.get('latency_ms', 0):.0f}ms"
        print(
            f"  {name:<28} "
            f"ADR={m['ADR']:.3f}  FPR={m['FPR']:.4f}  "
            f"F1={m['F1']:.3f}  lat={lat}"
        )


if __name__ == "__main__":
    main()
