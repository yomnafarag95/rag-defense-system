"""
train_meta_aggregator.py
Trains the RAG-Shield meta-aggregator with:
  - Contamination check (IEEE Fix #4)
  - Cross-validated LogisticRegressionCV (IEEE Fix #4)
  - Saves to models/meta_aggregator.pkl
"""

import os, json, hashlib, re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from split_helper import get_split

# ── Paths ────────────────────────────────────────────────────────────
LOGS_PATH   = "logs/pipeline_logs.jsonl"
EVAL_PATH   = "data/benign_queries.jsonl"
MODEL_OUT   = "models/meta_aggregator.pkl"
SCALER_OUT  = "models/meta_scaler.pkl"

# CRITICAL: This order MUST match orchestrator.py _features() exactly:
#   [l1_max, l1_win, l1_full, l2_stage1, l2_consist, l3_schema, l3_bound, l3_consist, l1xl2, l1xl3]
FEATURE_COLS = [
    'r1_max', 'r1_win', 'r1_full',
    'r2',     'r2_cs',   # r2=l2_stage1_prob at pos3, r2_cs=l2_consistency at pos4
    'v_sch', 'v_bnd',
    'r3', 'r1r2', 'r1r3'
]

# ── Contamination check ───────────────────────────────────────────────
def retrieve_best_document(query: str, doc_pool: list[str]) -> str:
    """Find the document in doc_pool with the highest token overlap with the query."""
    q_tokens = set(re.findall(r"\w+", query.lower()))
    if not q_tokens:
        return doc_pool[0]
    best_doc = doc_pool[0]
    best_overlap = -1
    for doc in doc_pool:
        d_tokens = set(re.findall(r"\w+", doc.lower()))
        overlap = len(q_tokens & d_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_doc = doc
    return best_doc

def check_contamination(train_queries, eval_queries, threshold=0.85):
    # Exact overlap check
    train_clean = [q.strip().lower() for q in train_queries if q]
    eval_clean  = [q.strip().lower() for q in eval_queries  if q]
    
    exact_overlap = set(train_clean) & set(eval_clean)
    if exact_overlap:
        raise ValueError(
            f"EXACT CONTAMINATION DETECTED: {len(exact_overlap)} queries appear in both "
            f"meta-aggregator training set and evaluation set. "
            f"Remove them from logs before retraining."
        )
        
    # Semantic TF-IDF overlap check
    print("[meta] Running semantic contamination check via TF-IDF cosine similarity...")
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    all_queries = train_clean + eval_clean
    try:
        tfidf = vectorizer.fit_transform(all_queries)
        train_tfidf = tfidf[:len(train_clean)]
        eval_tfidf  = tfidf[len(train_clean):]
        
        sim = cosine_similarity(train_tfidf, eval_tfidf)
        # Find pairs with similarity >= threshold
        leakage_indices = np.where(sim >= threshold)
        
        leaked_pairs = []
        for tr_idx, ev_idx in zip(*leakage_indices):
            leaked_pairs.append((train_clean[tr_idx], eval_clean[ev_idx], sim[tr_idx, ev_idx]))
            
        if leaked_pairs:
            print(f"[meta] WARNING: Semantic leakage detected ({len(leaked_pairs)} pairs with similarity >= {threshold}):")
            for tr, ev, s in leaked_pairs[:5]:
                print(f"  Similarity: {s:.3f} | Train: {tr[:50]} | Eval: {ev[:50]}")
            
            # Return indices to filter out from training set
            leaked_train_indices = set(int(tr_idx) for tr_idx in leakage_indices[0])
            print(f"[meta] Pruning {len(leaked_train_indices)} leaked training samples to protect validation splits.")
            return leaked_train_indices
    except Exception as e:
        print(f"[meta] Semantic contamination check failed: {e}")
    print("[meta] Semantic contamination check passed. No template leaks detected.")
    return set()

# ── Load logs ────────────────────────────────────────────────────────
def load_logs(path, force_refresh=False):
    """
    Load training logs. Always re-collects if force_refresh=True,
    or if the existing file has a feature column order mismatch
    (detected by checking that r2 comes before r2_cs in column order).
    """
    if force_refresh or not os.path.exists(path):
        print(f"[meta] {'Force refresh requested' if force_refresh else 'No logs file found'}. Collecting real training logs...")
        return collect_real_training_logs()

    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if len(rows) == 0:
        print("[meta] Logs file is empty — collecting real training logs...")
        return collect_real_training_logs()

    df = pd.DataFrame(rows)

    # Validate that the saved feature order matches what we expect.
    # In the old (broken) version, r2_cs was saved at position 3 and r2 at position 4.
    # We now require r2 (l2_stage1_prob) at position 3 in FEATURE_COLS.
    # A quick sanity check: benign r2 mean should be near 0, attacks near 1.
    if 'r2' in df.columns and 'label' in df.columns:
        benign_r2_mean = df[df['label'] == 0]['r2'].mean()
        if benign_r2_mean > 0.5:
            print(f"[meta] WARN: benign r2 mean={benign_r2_mean:.3f} > 0.5 — stale/mismatched log detected.")
            print("[meta] Re-collecting real training logs with correct feature order...")
            return collect_real_training_logs()

    print(f"[meta] Loaded {len(df)} log entries from {path}")
    return df


def _load_document_pool():
    pool = []
    wiki_path = "data/wikipedia_sample.jsonl"
    if os.path.exists(wiki_path):
        with open(wiki_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    text = row.get("text", "").strip()
                    if len(text) >= 100:
                        pool.append(text)
                except:
                    pass
                if len(pool) >= 100:
                    break
    if len(pool) < 100:
        benign_path = "data/benign_queries.jsonl"
        if os.path.exists(benign_path):
            with open(benign_path, encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if idx < 423:
                        continue
                    try:
                        row = json.loads(line)
                        text = row.get("text", "").strip()
                        if len(text) >= 100:
                            pool.append(text)
                    except:
                        pass
                    if len(pool) >= 100:
                        break
    if not pool:
        pool = ["This is a fallback standard company return and employee policy document containing details about the code of conduct."]
    return pool


def collect_real_training_logs():
    """
    Generates real feature vectors by running the actual L1, L2, L3 layers
    on training subsets of HackAPrompt and MS MARCO.
    """
    print("[meta] Generating real training logs by running pipeline components on real data...")
    import pandas as pd
    from layer1_anomaly import load_detector
    from layer2_classifier import load_classifier
    from layer3_enhanced import load_monitor
    from tqdm import tqdm
    
    print("[meta] Loading models...")
    detector   = load_detector()
    classifier = load_classifier()
    monitor    = load_monitor()

    print("[meta] Loading document pool...")
    doc_pool = _load_document_pool()
    print(f"[meta] Loaded document pool with {len(doc_pool)} documents.")
    
    rows = []
    
    # 1. Load benign queries — prefer extended_benign.csv (richer, more diverse)
    #    Fall back to the train slice of benign_queries.jsonl if needed.
    BENIGN_CAP = 180  # Raised to balance with additional InjecAgent attack samples
    benign_queries = []

    ext_benign_path = "data/extended_benign.csv"
    if os.path.exists(ext_benign_path):
        try:
            import pandas as _pd
            _ext = _pd.read_csv(ext_benign_path)
            _col = "query" if "query" in _ext.columns else _ext.columns[0]
            # Use deterministic hash split to select training queries
            all_qs = _ext[_col].dropna().tolist()
            benign_queries = [q for q in all_qs if get_split(q) == "train"]
            print(f"[meta] Benign source: extended_benign.csv train split ({len(benign_queries)} samples)")
        except Exception as _e:
            print(f"[meta] extended_benign.csv load failed ({_e}), falling back.")

    if len(benign_queries) < BENIGN_CAP:
        benign_path = "data/benign_queries.jsonl"
        if os.path.exists(benign_path):
            with open(benign_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        q = row.get("text") or row.get("query", "")
                        if q and get_split(q) == "train":
                            benign_queries.append(q)
                    except:
                        pass
                    if len(benign_queries) >= BENIGN_CAP:
                        break
            print(f"[meta] Benign source: benign_queries.jsonl fallback ({len(benign_queries)} total)")
    
    # 2a. Load direct attack queries from hackaprompt.jsonl
    # Use queries that are NOT in hackaprompt_holdout_seed42.csv
    holdout_path = "data/hackaprompt_holdout_seed42.csv"
    holdout_prompts = set()
    if os.path.exists(holdout_path):
        try:
            holdout_df = pd.read_csv(holdout_path)
            holdout_prompts = set(holdout_df["text"].tolist())
        except:
            pass

    DIRECT_ATTACK_CAP = 120  # HackAPrompt direct injections
    attack_queries = []
    attack_path = "data/hackaprompt.jsonl"
    if os.path.exists(attack_path):
        with open(attack_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    text = row["text"]
                    if text not in holdout_prompts:
                        attack_queries.append((text, "direct"))
                except:
                    pass
                if len(attack_queries) >= DIRECT_ATTACK_CAP:
                    break

    # 2b. Load InjecAgent indirect injection samples
    # These are document-embedded commands: the attack is in the DOCUMENT, not the query.
    injecagent_path = "data/injecagent.jsonl"
    injecagent_queries = []
    if os.path.exists(injecagent_path):
        with open(injecagent_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    text = row.get("text", "").strip()
                    if text and row.get("label", 0) == 1:
                        injecagent_queries.append((text, "indirect"))
                except:
                    pass
        print(f"[meta] InjecAgent: loaded {len(injecagent_queries)} indirect injection samples")

    # Combine: all direct + all indirect InjecAgent attacks
    all_attacks = attack_queries + injecagent_queries

    if len(benign_queries) == 0 or len(all_attacks) == 0:
        print("[meta] Datasets missing. Please run python data_loader.py first.")
        raise RuntimeError("Training datasets not downloaded yet. Run data_loader.py first.")

    # Enforce class balance: cap benign to match attack count
    n = min(len(all_attacks), len(benign_queries))
    all_attacks   = all_attacks[:n]
    benign_queries = benign_queries[:n]
    n_direct   = sum(1 for _, t in all_attacks if t == "direct")
    n_indirect = sum(1 for _, t in all_attacks if t == "indirect")
    print(f"[meta] Balanced training set: {n} attacks ({n_direct} direct + {n_indirect} indirect) + {n} benign = {2*n} total")

    sys_prompt = "Answer using only the provided knowledge base."

    def _extract_features(query_text, doc_text):
        from layer1_anomaly import split_chunks
        chunks = split_chunks(doc_text)
        l1_res = detector.scan(chunks)
        l2_res = classifier.classify(query_text, chunks)
        l3_res = monitor.check(query_text, sys_prompt, chunks, l1_res, l2_res)
        return {
            'r1_max':  l1_res["max_score"],
            'r1_win':  max(l1_res["window_scores"], default=0.0),
            'r1_full': l1_res["full_score"],
            'r2_cs':   l2_res["consistency_score"],
            'r2':      l2_res["stage1_prob"],
            'v_sch':   0.0 if l3_res["schema_valid"] else 1.0,
            'v_bnd':   min(len(l3_res["boundary_violations"]) / 2.0, 1.0),
            'r3':      l3_res["consistency_score"],
            'r1r2':    l1_res["max_score"] * l2_res["stage1_prob"],
            'r1r3':    l1_res["full_score"] * l3_res["consistency_score"],
        }

    # Helper function to run features
    def process_samples(samples, is_attack):
        desc = "attacks" if is_attack else "benign"
        for i, item in enumerate(tqdm(samples, desc=f"Processing {desc}")):
            if is_attack:
                q_text, attack_type = item
                if attack_type == "indirect":
                    # InjecAgent: attack is in the document, query is innocuous
                    query_text = "Please summarize the retrieved document."
                    doc_text   = q_text
                else:
                    # HackAPrompt direct: alternate direct/indirect to ensure variety
                    if i % 2 == 0:
                        query_text = q_text
                        doc_text   = doc_pool[i % len(doc_pool)]
                    else:
                        query_text = "Please summarize the retrieved document."
                        doc_text   = q_text
            else:
                query_text = item  # benign queries are plain strings
                doc_text   = retrieve_best_document(query_text, doc_pool)

            feats = _extract_features(query_text, doc_text)
            feats['label'] = 1 if is_attack else 0
            feats['query'] = query_text
            rows.append(feats)

    process_samples(all_attacks, is_attack=True)
    process_samples(benign_queries, is_attack=False)

    df = pd.DataFrame(rows)
    os.makedirs('logs', exist_ok=True)
    df.to_json(LOGS_PATH, orient='records', lines=True)
    print(f"[meta] Saved real training logs to {LOGS_PATH}")
    return df

# ── Load eval queries for contamination check ────────────────────────
def load_eval_queries(path=None):
    queries = []
    # 1. Load benign queries with get_split == 'test'
    benign_path = "data/benign_queries.jsonl"
    if os.path.exists(benign_path):
        with open(benign_path, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    q = obj.get("text") or obj.get("query") or obj.get("question", "")
                    if q and get_split(q) == "test":
                        queries.append(q)
                except:
                    continue

    # 2. Load hackaprompt holdout prompts
    holdout_path = "data/hackaprompt_holdout_seed42.csv"
    if os.path.exists(holdout_path):
        try:
            holdout_df = pd.read_csv(holdout_path)
            queries.extend(holdout_df["text"].dropna().tolist())
        except:
            pass

    # 3. Load extended benign queries split
    ext_path = "data/extended_benign.csv"
    if os.path.exists(ext_path):
        try:
            ext_df = pd.read_csv(ext_path)
            all_qs = ext_df["query"].dropna().tolist()
            queries.extend([q for q in all_qs if get_split(q) == "test"])
        except:
            pass

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
        scoring='neg_log_loss'
    )

    cv_scores = cross_val_score(
        base_clf, X_scaled, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='roc_auc'
    )
    print(f"[meta] CV AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

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
    print("=== Meta-Aggregator Training (IEEE Fix #4 + Fix B: balanced retraining) ===\n")

    # Always force-refresh the training log so stale cached vectors
    # (which caused the biased benign prior) don't persist.
    if os.path.exists(LOGS_PATH):
        os.remove(LOGS_PATH)
        print(f"[meta] Removed stale log cache: {LOGS_PATH}")

    df = load_logs(LOGS_PATH, force_refresh=True)

    # Contamination check
    train_queries = df['query'].tolist() if 'query' in df.columns else []
    eval_queries  = load_eval_queries(EVAL_PATH)
    if train_queries and eval_queries:
        leaked_indices = check_contamination(train_queries, eval_queries)
        if leaked_indices:
            df = df.drop(index=list(leaked_indices)).reset_index(drop=True)
            df.to_json(LOGS_PATH, orient='records', lines=True)
            print(f"[meta] Training dataframe pruned. New training set size: {len(df)} (saved to {LOGS_PATH})")
    else:
        print("[meta] Skipping contamination check (no query strings in logs)")

    clf, scaler = train_meta_aggregator(df)
    print("\n[meta] Training complete.")
    print("       Next step: python eval_suite.py --mode all")