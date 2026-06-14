"""
diagnose_fpr.py
Diagnoses why 100% of benign queries are being flagged.
Run with: .venv311\Scripts\python.exe diagnose_fpr.py
"""
import json
import statistics

# ── 1. Parse eval_results.jsonl ──────────────────────────────────────────────
benign = []
attack = []
with open("logs/eval_results.jsonl", encoding="utf-8") as f:
    for line in f:
        try:
            r = json.loads(line)
            if r["true_label"] == 0:
                benign.append(r)
            else:
                attack.append(r)
        except Exception:
            pass

print(f"\n{'='*60}")
print(f"Total benign : {len(benign)}")
print(f"Total attack : {len(attack)}")

# ── 2. Action distribution for benign ────────────────────────────────────────
actions = {}
for r in benign:
    a = r.get("action", "?")
    actions[a] = actions.get(a, 0) + 1
print(f"\nBenign action distribution: {actions}")

# ── 3. Score distributions for benign ────────────────────────────────────────
l1 = [r["l1_score"]   for r in benign]
l2 = [r["l2_score"]   for r in benign]
rs = [r["risk_score"] for r in benign]

def stats(name, vals):
    print(f"  {name:12s} mean={statistics.mean(vals):.4f}  "
          f"median={statistics.median(vals):.4f}  "
          f"min={min(vals):.4f}  max={max(vals):.4f}")

print("\nBenign score distributions:")
stats("L1 score",   l1)
stats("L2 score",   l2)
stats("Risk score", rs)

# ── 4. Show 10 blocked benign samples ─────────────────────────────────────────
blocked_benign = [r for r in benign if r.get("action") in ("blocked", "hard_block")]
print(f"\nBlocked benign count: {len(blocked_benign)}")
print("\nSample blocked-benign entries:")
for r in blocked_benign[:10]:
    print(f"  query  : {r['query'][:70]!r}")
    print(f"  doc    : {r['text'][:70]!r}")
    print(f"  l1={r['l1_score']:.4f}  l2={r['l2_score']:.4f}  risk={r['risk_score']:.4f}  action={r['action']}")
    print()

# ── 5. L2 score histogram buckets ─────────────────────────────────────────────
buckets = {"0.0-0.3": 0, "0.3-0.5": 0, "0.5-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
for v in l2:
    if v < 0.3:      buckets["0.0-0.3"] += 1
    elif v < 0.5:    buckets["0.3-0.5"] += 1
    elif v < 0.6:    buckets["0.5-0.6"] += 1
    elif v < 0.8:    buckets["0.6-0.8"] += 1
    else:            buckets["0.8-1.0"] += 1
print("L2 score histogram (benign):")
for k, v in buckets.items():
    bar = "#" * (v // 5)
    print(f"  {k}: {v:4d}  {bar}")

# ── 6. Check if the problem is doc_score vs query score ───────────────────────
print("\n--- Now testing DeBERTa directly on benign doc text ---")
benign_doc = "This document contains standard company information regarding employee benefits and policies."
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok   = AutoTokenizer.from_pretrained("protectai/deberta-v3-base-prompt-injection-v2")
    model = AutoModelForSequenceClassification.from_pretrained(
        "protectai/deberta-v3-base-prompt-injection-v2"
    )
    model.eval()
    test_strings = [
        benign_doc,
        "What are the office hours?",
        "why do nails get rusty",
        "who is the CEO of the company?",
    ]
    for s in test_strings:
        inputs = tok(s, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
        print(f"  prob={float(probs[1]):.4f}  text={s[:65]!r}")
except Exception as e:
    print(f"  Could not run DeBERTa directly: {e}")

print("\nDone.")
