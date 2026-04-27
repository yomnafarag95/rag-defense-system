"""
orchestrator.py
───────────────
Meta Aggregator + Pipeline Orchestrator

Responsibilities
────────────────
  1. Runs all three layers in sequence
  2. Collects evidence vectors from each layer
  3. Passes evidence to the learned meta-aggregator
  4. Returns a single unified result dict to app.py
  5. Logs every run to logs/pipeline.jsonl

Meta Aggregator
───────────────
  Model: calibrated LogisticRegressionCV (sklearn)
  Features: 10 signals (layer scores + cross-layer interaction terms)
  Output: calibrated attack probability + action label

Wire into app.py
────────────────
  from orchestrator import run_pipeline

  result = run_pipeline(
      document      = doc_input,
      query         = query_input,
      system_prompt = sys_input,
  )
  st.session_state.results = result
"""

import json
import pickle
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.calibration import CalibratedClassifierCV

from config import (
    META_MODEL_PATH,
    META_HARD_BLOCK_SINGLE,
    META_HARD_BLOCK_VIOLS,
    META_BLOCK_THRESHOLD,
    META_MONITOR_THRESHOLD,
    LOG_PATH,
    MAX_HISTORY_ITEMS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Meta Aggregator
# ─────────────────────────────────────────────────────────────────────────────

class MetaAggregator:
    """
    Learned meta-classifier that combines evidence from all three layers.

    Why learned instead of hand-weighted
    ─────────────────────────────────────
    Hand-picked weights assume we know which layer matters most.
    LogisticRegressionCV learns the optimal weights from real blocked/passed
    logs. The calibration wrapper ensures the output probabilities are
    statistically meaningful (not just rankings).

    Feature vector (10 dimensions)
    ───────────────────────────────
      l1_max_chunk   : worst individual chunk anomaly score
      l1_window_max  : worst sliding-window score (payload splitting signal)
      l1_full        : full-document anomaly score
      l2_stage1      : Layer 2 attack probability
      l2_consistency : query-document lexical consistency
      l3_schema      : 0 if schema valid, 1 if violated
      l3_boundary    : normalised boundary violation count (capped at 1)
      l3_consistency : Layer 3 response consistency risk
      l1xl2          : interaction term (l1_max x l2_stage1)
      l1xl3          : interaction term (l1_full x l3_consistency)

    Fallback weights (before model is trained on real logs)
    ────────────────────────────────────────────────────────
      L2 carries 60% weight — it is the strongest signal before fine-tuning.
      A detected attack (L2 ~ 0.82) scores ~0.52, crossing META_BLOCK_THRESHOLD.
      A clean query  (L2 ~ 0.04) scores ~0.03, safely below threshold.
      XR-500 false positive (L1=1.0, L2=0.04) scores ~0.11 — ALLOW.
    """

    def __init__(self):
        base  = LogisticRegressionCV(Cs=10, cv=5, penalty="l2",
                                     scoring="roc_auc", max_iter=1000)
        self.model = CalibratedClassifierCV(base, cv=5)
        self._fitted = False

    def _features(self, l1: dict, l2: dict, l3: dict) -> np.ndarray:
        f = np.array([
            l1["max_score"],
            max(l1["window_scores"], default=0.0),
            l1["full_score"],
            l2["stage1_prob"],
            l2["consistency_score"],
            0.0 if l3["schema_valid"] else 1.0,
            min(len(l3["boundary_violations"]) / max(META_HARD_BLOCK_VIOLS, 1), 1.0),
            l3["consistency_score"],
            l1["max_score"] * l2["stage1_prob"],        # interaction L1 x L2
            l1["full_score"] * l3["consistency_score"], # interaction L1 x L3
        ])
        return f  # 1D array — reshaped to (1,-1) only when passed to sklearn

    def fit(self, feature_matrix: np.ndarray, labels: list) -> "MetaAggregator":
        """
        Train on historical blocked/passed runs.
        labels: 1 = attack confirmed, 0 = benign confirmed (from human review)
        """
        self.model.fit(feature_matrix, labels)
        self._fitted = True
        return self

    def predict(self, l1: dict, l2: dict, l3: dict) -> dict:
        features = self._features(l1, l2, l3)

        # Hard escalation — only boundary violations trigger instant block.
        # Single-layer scores do NOT hard block to prevent L1 false positives
        # on technical documents (network specs, product manuals, etc.)
        hard_block = (
            len(l3["boundary_violations"]) >= META_HARD_BLOCK_VIOLS
        )

        if hard_block:
            return {
                "risk_score":  1.0,
                "action":      "hard_block",
                "confidence":  1.0,
                "hard_block":  True,
                "features":    features.tolist(),
            }

        if self._fitted:
            prob = float(self.model.predict_proba(features.reshape(1, -1))[0][1])
        else:
            # Fallback weighted sum — used before model is trained on real logs.
            # L2 weight = 0.60 so a detected attack (L2~0.82) scores ~0.52,
            # crossing META_BLOCK_THRESHOLD (0.45) and triggering BLOCKED.
            w = np.array([0.08, 0.04, 0.03, 0.60, 0.04,
                          0.08, 0.06, 0.04, 0.02, 0.01])
            prob = float(np.clip(float(np.dot(features, w)), 0, 1))

        prob   = round(prob, 4)
        action = (
            "blocked" if prob > META_BLOCK_THRESHOLD   else
            "monitor" if prob > META_MONITOR_THRESHOLD else
            "allow"
        )
        confidence = round(abs(prob - 0.5) * 2, 4)

        return {
            "risk_score":  prob,
            "action":      action,
            "confidence":  confidence,
            "hard_block":  False,
            "features":    features.tolist(),
        }

    def save(self, path: str = META_MODEL_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"MetaAggregator saved -> {path}")

    @classmethod
    def load(cls, path: str = META_MODEL_PATH) -> "MetaAggregator":
        if not Path(path).exists():
            logger.warning(f"MetaAggregator model not found at {path}. Using fallback weights.")
            return cls()
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline run logger
# ─────────────────────────────────────────────────────────────────────────────

class PipelineLogger:
    """
    Appends every pipeline run to a JSONL log file.
    Each entry contains full evidence vectors for later meta-model retraining.

    confirmed_attack field is filled by human reviewers
    via a separate review script — not by the pipeline itself.
    """

    def __init__(self, path: str = LOG_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, result: dict) -> None:
        entry = {
            "timestamp":        datetime.utcnow().isoformat(),
            "query_hash":       hash(result.get("query_short", "")),
            "action":           result["action"],
            "risk_score":       result["meta"]["risk_score"],
            "l1_max":           result["l1"]["max_score"],
            "l2_prob":          result["l2"]["stage1_prob"],
            "l3_cs":            result["l3"]["consistency_score"],
            "attack_type":      result["l2"]["stage2_label"],
            "features":         result["meta"]["features"],
            "confirmed_attack": None,  # filled by human reviewer
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def load_confirmed(self) -> tuple:
        """Load confirmed labels for meta-model retraining."""
        X, y = [], []
        if not self.path.exists():
            return np.array(X), y
        with open(self.path) as f:
            for line in f:
                row = json.loads(line)
                if row["confirmed_attack"] is not None:
                    X.append(row["features"])
                    y.append(int(row["confirmed_attack"]))
        return np.array(X), y


# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline — the single function called by app.py
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(document:       str,
                 query:          str,
                 system_prompt:  str,
                 l1_detector=None,
                 l2_classifier=None,
                 l3_monitor=None,
                 meta_aggregator=None,
                 raw_response:   Optional[str] = None) -> dict:
    """
    Full pipeline execution.

    Parameters
    ----------
    document        : raw KB document text (will be chunked internally)
    query           : user query string
    system_prompt   : system prompt string
    l1_detector     : AnomalyDetector instance (from layer1_anomaly.py)
    l2_classifier   : IntentClassifier instance (from layer2_classifier.py)
    l3_monitor      : BehavioralMonitor instance (from layer3_semantic.py)
    meta_aggregator : MetaAggregator instance (from this file)
    raw_response    : optional LLM response string (for Layer 3 output check)

    Returns
    -------
    dict with keys matching app.py's st.session_state.results schema:
        chunks, l1, l2, l3, meta,
        blocking_layer, blocked, monitored, action,
        timestamp, query_short
    """
    from layer1_anomaly import split_chunks

    # Chunk document
    chunks = split_chunks(document)

    # Layer 1
    if l1_detector is not None:
        l1 = l1_detector.scan(chunks)
    else:
        raise RuntimeError(
            "l1_detector is None. Load with load_detector() and pass to run_pipeline()."
        )

    # Layer 2
    if l2_classifier is not None:
        l2 = l2_classifier.classify(query, chunks)
    else:
        raise RuntimeError(
            "l2_classifier is None. Load with load_classifier() and pass to run_pipeline()."
        )

    # Layer 3
    if l3_monitor is not None:
        l3 = l3_monitor.check(query, system_prompt, chunks, l1, l2, raw_response)
    else:
        raise RuntimeError(
            "l3_monitor is None. Load with load_monitor() and pass to run_pipeline()."
        )

    # Meta aggregator
    agg  = meta_aggregator or MetaAggregator()
    meta = agg.predict(l1, l2, l3)

    # Determine which layer blocked
    blocking_layer = None
    if l1["blocked"]:
        blocking_layer = "Layer 1 - Anomaly Detection"
    elif l2["blocked"]:
        blocking_layer = "Layer 2 - Intent Classifier"
    elif l3["blocked"]:
        blocking_layer = "Layer 3 - Behavioral Monitor"
    elif meta["action"] in ("blocked", "hard_block"):
        blocking_layer = "Meta Aggregator - Combined Risk"

    final_blocked   = meta["action"] in ("blocked", "hard_block")
    final_monitored = meta["action"] == "monitor"

    result = {
        "chunks":         chunks,
        "l1":             l1,
        "l2":             l2,
        "l3":             l3,
        "meta":           meta,
        "blocking_layer": blocking_layer,
        "blocked":        final_blocked,
        "monitored":      final_monitored,
        "action":         meta["action"],
        "timestamp":      datetime.utcnow().strftime("%H:%M:%S UTC"),
        "query_short":    query[:60] + ("..." if len(query) > 60 else ""),
    }

    # Log
    try:
        PipelineLogger().log(result)
    except Exception as e:
        logger.warning(f"Logging failed: {e}")

    logger.info(
        f"action={meta['action']}  risk={meta['risk_score']:.4f}  "
        f"l1={l1['max_score']:.4f}  l2={l2['stage1_prob']:.4f}  "
        f"l3={l3['consistency_score']:.4f}"
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Meta-model retraining  (python orchestrator.py retrain)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or sys.argv[1] != "retrain":
        print("Usage: python orchestrator.py retrain")
        print("       Retrains the meta-aggregator on confirmed log labels.")
        sys.exit(0)

    print("\n=== Meta Aggregator - Retraining ===\n")

    pl = PipelineLogger()
    X, y = pl.load_confirmed()

    if len(X) == 0:
        print("[meta] No confirmed labels found in logs.")
        print("       Label entries in logs/pipeline.jsonl by setting confirmed_attack=true/false.")
        sys.exit(0)

    print(f"[meta] Found {len(y)} confirmed examples.")
    print(f"       Attacks: {sum(y)}  Benign: {len(y)-sum(y)}")

    agg = MetaAggregator()
    agg.fit(X, y)
    agg.save()
    print(f"[meta] Retraining complete. Model saved to {META_MODEL_PATH}")