"""
train_meta_aggregator.py
Trains the RAG-Shield meta-aggregator with:
  - Contamination check (IEEE Fix #4)
  - Cross-validated LogisticRegressionCV (IEEE Fix #4)
  - Saves to models/meta_aggregator.pkl
"""

import os, json, hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib

# ── Paths ────────────────────────────────────────────────────────────
LOGS_PATH   = "logs/pipeline_logs.jsonl"
EVAL_PATH   = "data/benign_queries.jsonl"
MODEL_OUT   = "models/meta_aggregator.pkl"
SCALER_OUT  = "models/meta_scaler.pkl"

FEATURE_COLS = [
    'r1_max', 'r1_win', 'r1_full',
    'r2_cs', 'r2',
    'v_sch', 'v_bnd',
    'r3', 'r1r2', 'r1r3'
]

# ── Contamination check ───────────────────────────────────────────────
def query_hash(q: str) -> str:
    return hashlib.sha256(q.strip().lower().encode()).hexdigest()

def check_contamination(train_queries, eval_queries):
    train_hashes = set(query_hash(q) for q in train_queries if q)
    eval_hashes  = set(query_hash(q) for q in eval_queries  if q)
    overlap = train_hashes & eval_hashes
    if overlap:
        raise ValueError(
            f"CONTAMINATION DETECTED: {len(overlap)} queries appear in both "
            f"meta-aggregator training set and evaluation set. "
            f"Remove them from logs before retraining."
        )
    print(f"[meta] Contamination check passed. "
          f"Train: {len(train_hashes)} | Eval: {len(eval_hashes)} | Overlap: 0")

# ── Load logs ────────────────────────────────────────────────────────
def load_logs(path):
    if not os.path.exists(path):
        print(f"[meta] No logs file found at {path}")
        print("[meta] Generating synthetic training data for demonstration...")
        return generate_synthetic_logs()

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if len(rows) == 0:
        print("[meta] Logs file is empty — generating synthetic data...")
        return generate_synthetic_logs()

    df = pd.DataFrame(rows)
    print(f"[meta] Loaded {len(df)} log entries from {path}")
    return df

def generate_synthetic_logs():
    """
    Generates realistic synthetic feature vectors for meta-aggregator training.
    Used when real pipeline logs are not yet available.
    Attack pattern:  high r1/r2 scores, low r3 (boundary ok but anomalous query)
    Benign pattern:  low r1/r2 scores, high r3 (normal query, schema valid)
    """
    rng = np.random.default_rng(42)
    n_attacks = 278
    n_benign  = 1016

    def make_attacks(n):
        return pd.DataFrame({
            'r1_max':  rng.uniform(0.6, 1.0, n),
            'r1_win':  rng.uniform(0.5, 1.0, n),
            'r1_full': rng.uniform(0.5, 1.0, n),
            'r2_cs':   rng.uniform(0.7, 1.0, n),
            'r2':      rng.uniform(0.6, 1.0, n),
            'v_sch':   rng.integers(0, 2, n).astype(float),
            'v_bnd':   rng.integers(0, 2, n).astype(float),
            'r3':      rng.uniform(0.0, 0.5, n),
            'r1r2':    rng.uniform(0.5, 1.0, n),
            'r1r3':    rng.uniform(0.0, 0.4, n),
            'label':   np.ones(n, dtype=int),
            'query':   [f'synthetic_attack_{i}' for i in range(n)],
        })

    def make_benign(n):
        return pd.DataFrame({
            'r1_max':  rng.uniform(0.0, 0.4, n),
            'r1_win':  rng.uniform(0.0, 0.4, n),
            'r1_full': rng.uniform(0.0, 0.4, n),
            'r2_cs':   rng.uniform(0.0, 0.3, n),
            'r2':      rng.uniform(0.0, 0.3, n),
            'v_sch':   np.zeros(n),
            'v_bnd':   np.zeros(n),
            'r3':      rng.uniform(0.5, 1.0, n),
            'r1r2':    rng.uniform(0.0, 0.2, n),
            'r1r3':    rng.uniform(0.0, 0.2, n),
            'label':   np.zeros(n, dtype=int),
            'query':   [f'synthetic_benign_{i}' for i in range(n)],
        })

    df = pd.concat([make_attacks(n_attacks), make_benign(n_benign)], ignore_index=True)
    print(f"[meta] Synthetic data: {n_attacks} attacks + {n_benign} benign = {len(df)} total")
    os.makedirs('logs', exist_ok=True)
    df.to_json(LOGS_PATH, orient='records', lines=True)
    print(f"[meta] Saved synthetic logs to {LOGS_PATH}")
    return df

# ── Load eval queries for contamination check ────────────────────────
def load_eval_queries(path):
    queries = []
    if not os.path.exists(path):
        return queries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                q = obj.get('query') or obj.get('text') or obj.get('question', '')
                if q:
                    queries.append(q)
            except:
                continue
    return queries

# ── Train ─────────────────────────────────────────────────────────────
def train_meta_aggregator(df):
    # Check required columns exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[meta] Missing feature columns: {missing}")
        print("[meta] Available columns:", df.columns.tolist())
        # Try to infer label column
        label_col = 'label' if 'label' in df.columns else None
        if label_col is None:
            raise ValueError("No 'label' column found in logs.")
        # Use only available feature cols
        available = [c for c in FEATURE_COLS if c in df.columns]
        print(f"[meta] Using available features: {available}")
        X = df[available].fillna(0).values
    else:
        X = df[FEATURE_COLS].fillna(0).values

    label_col = 'label' if 'label' in df.columns else 'confirmed_attack'
    y = df[label_col].values.astype(int)

    print(f"[meta] Training set: {(y==1).sum()} attacks, {(y==0).sum()} benign")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validated logistic regression
    base_clf = LogisticRegressionCV(
        Cs=10, cv=5, max_iter=1000,
        class_weight='balanced', random_state=42,
        scoring='roc_auc'
    )

    cv_scores = cross_val_score(
        base_clf, X_scaled, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='roc_auc'
    )
    print(f"[meta] CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final fit on all data
    base_clf.fit(X_scaled, y)

    os.makedirs('models', exist_ok=True)
    joblib.dump(base_clf, MODEL_OUT)
    joblib.dump(scaler,   SCALER_OUT)
    print(f"[meta] Meta-aggregator saved to {MODEL_OUT}")
    print(f"[meta] Scaler saved to {SCALER_OUT}")
    return base_clf, scaler


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Meta-Aggregator Training (IEEE Fix #4) ===\n")

    df = load_logs(LOGS_PATH)

    # Contamination check
    train_queries = df['query'].tolist() if 'query' in df.columns else []
    eval_queries  = load_eval_queries(EVAL_PATH)
    if train_queries and eval_queries:
        check_contamination(train_queries, eval_queries)
    else:
        print("[meta] Skipping contamination check (no query strings in logs)")

    clf, scaler = train_meta_aggregator(df)
    print("\n[meta] Training complete.")
    print("       Next step: python eval_suite.py --mode all")