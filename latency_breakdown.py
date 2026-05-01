import time
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline as hf_pipeline
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

N_WARMUP = 3
N_TRIALS = 20
QUERY = "What is our company refund policy for enterprise clients?"
PROMPT = "You are a helpful enterprise assistant."

def measure(fn, n_warmup=N_WARMUP, n_trials=N_TRIALS):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.mean(times), np.percentile(times, 95)

print("Loading models (this takes ~30s) ...")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
l2 = hf_pipeline("text-classification", model="protectai/deberta-v3-base-prompt-injection-v2", device=-1)
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

iso = IsolationForest(n_estimators=100, random_state=42)
dummy_embs = embedder.encode(["clean document number " + str(i) for i in range(50)])
iso.fit(dummy_embs)

meta = LogisticRegression()
meta.fit([[0]*10, [1]*10], [0, 1])
x_dummy = np.random.rand(1, 10)

print(f"Profiling ({N_TRIALS} trials, {N_WARMUP} warmup) ...\n")

emb_m, emb_p = measure(lambda: embedder.encode(QUERY))

emb_vec = embedder.encode(QUERY).reshape(1, -1)
l1_m, l1_p = measure(lambda: iso.score_samples(emb_vec))

l2_m, l2_p = measure(lambda: l2(QUERY, truncation=True))

l3_m, l3_p = measure(lambda: ce.predict([(QUERY, PROMPT)]))

meta_m, meta_p = measure(lambda: meta.predict_proba(x_dummy))

def full():
    e = embedder.encode(QUERY).reshape(1, -1)
    iso.score_samples(e)
    l2(QUERY, truncation=True)
    ce.predict([(QUERY, PROMPT)])
    meta.predict_proba(x_dummy)

full_m, full_p = measure(full)

def early_exit():
    e = embedder.encode(QUERY).reshape(1, -1)
    iso.score_samples(e)
    meta.predict_proba(x_dummy)

exit_m, exit_p = measure(early_exit)

print("=" * 58)
print("  TABLE V: Per-Component Latency Breakdown")
print("=" * 58)
print(f"  {'Component':<32} {'Mean(ms)':>9} {'P95(ms)':>9}")
print("-" * 58)

rows = [
    ("Embedding (MiniLM-L6-v2)", emb_m, emb_p),
    ("Layer 1 Ensemble", l1_m, l1_p),
    ("Layer 2 DeBERTa", l2_m, l2_p),
    ("Layer 3 Cross-Encoder", l3_m, l3_p),
    ("Meta-Aggregator", meta_m, meta_p),
]
for name, m, p in rows:
    print(f"  {name:<32} {m:>9.1f} {p:>9.1f}")

print("-" * 58)
print(f"  {'L1 early-exit (emb+L1+meta)':<32} {exit_m:>9.1f} {exit_p:>9.1f}")
print(f"  {'Full pipeline (all layers)':<32} {full_m:>9.1f} {full_p:>9.1f}")
print("=" * 58)

paper_latency = 346.5
print(f"\n  Paper claims: {paper_latency} ms")
print(f"  Measured full pipeline: {full_m:.1f} ms")
print(f"  Difference: {full_m - paper_latency:+.1f} ms")

if full_m > 500:
    print("\n  WARNING: Latency > 500ms. Possible causes:")
    print("    - Running on CPU (paper may have used GPU)")
    print("    - Cold cache / first-run overhead")
    print("    - DeBERTa not quantized (INT8 can save ~50%)")