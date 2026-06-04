"""
orchestrator.py
───────────────
Meta Aggregator + Pipeline Orchestrator

Responsibilities
────────────────
  1. Runs all three layers in sequence
  2. Collects evidence vectors from each layer
  3. Passes evidence to the learned meta-aggregator
  4. Returns a single unified result dict to app.py / eval_suite.py
  5. Logs every run to logs/pipeline.jsonl

Meta Aggregator
───────────────
  Model: calibrated LogisticRegressionCV (sklearn)
  Features: 10 signals (layer scores + cross-layer interaction terms)
  Output: calibrated attack probability + action label
"""

import json
import pickle
import logging
import joblib
import os
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
    ENABLE_L1_EARLY_EXIT,
)

from keyword_detector import keyword_check

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

    Feature vector (10 dimensions)
    ───────────────────────────────
      l1_max_chunk   : worst individual chunk anomaly score
      l1_window_max  : worst sliding-window score
      l1_full        : full-document anomaly score
      l2_stage1      : Layer 2 attack probability
      l2_consistency : query-document lexical consistency
      l3_schema      : 0 if schema valid, 1 if violated
      l3_boundary    : normalised boundary violation count
      l3_consistency : Layer 3 response consistency risk
      l1xl2          : interaction term (l1_max x l2_stage1)
      l1xl3          : interaction term (l1_full x l3_consistency)
    """

    def __init__(self, model=None, scaler=None):
        if model is None:
            base = LogisticRegressionCV(
                Cs=10,
                cv=5,
                penalty="l2",
                scoring="roc_auc",
                max_iter=1000,
            )
            self.model = CalibratedClassifierCV(base, cv=5)
            self._fitted = False
        else:
            self.model = model
            self._fitted = True

        self.scaler = scaler

    def _features(self, l1: dict, l2: dict, l3: dict) -> np.ndarray:
        return np.array([
            l1["max_score"],
            max(l1["window_scores"], default=0.0),
            l1["full_score"],
            l2["stage1_prob"],
            l2["consistency_score"],
            0.0 if l3["schema_valid"] else 1.0,
            min(len(l3["boundary_violations"]) / max(META_HARD_BLOCK_VIOLS, 1), 1.0),
            l3["consistency_score"],
            l1["max_score"] * l2["stage1_prob"],
            l1["full_score"] * l3["consistency_score"],
        ])

    def fit(self, feature_matrix: np.ndarray, labels: list) -> "MetaAggregator":
        if self.scaler is not None:
            feature_matrix = self.scaler.fit_transform(feature_matrix)
        self.model.fit(feature_matrix, labels)
        self._fitted = True
        return self

    def predict(self, l1: dict, l2: dict, l3: dict, query: str = "") -> dict:
        features = self._features(l1, l2, l3)

        # Hard escalation — boundary violations trigger instant block.
        hard_block = len(l3["boundary_violations"]) >= META_HARD_BLOCK_VIOLS

        if hard_block:
            return {
                "risk_score": 1.0,
                "action": "hard_block",
                "confidence": 1.0,
                "hard_block": True,
                "features": features.tolist(),
                "keyword_boost_applied": False,
                "keyword_match": None,
                "keyword_boost": 0.0,
            }

        features_for_model = features.reshape(1, -1)
        if self.scaler is not None:
            features_for_model = self.scaler.transform(features_for_model)

        if self._fitted:
            prob = float(self.model.predict_proba(features_for_model)[0][1])
        else:
            # Fallback weighted sum
            w = np.array([0.08, 0.04, 0.03, 0.60, 0.04,
                          0.08, 0.06, 0.04, 0.02, 0.01])
            prob = float(np.clip(float(np.dot(features, w)), 0, 1))

        kw_found = False
        kw_match = None
        kw_boost = 0.0

        if query:
            kw_found, kw_match, kw_boost = keyword_check(query)
            if kw_found:
                prob = min(prob + kw_boost, 1.0)
                logger.info(f"Keyword boost: '{kw_match}' +{kw_boost:.2f} -> {prob:.4f}")

        prob = round(prob, 4)
        action = (
            "blocked" if prob > META_BLOCK_THRESHOLD else
            "monitor" if prob > META_MONITOR_THRESHOLD else
            "allow"
        )
        confidence = round(abs(prob - 0.5) * 2, 4)

        return {
            "risk_score": prob,
            "action": action,
            "confidence": confidence,
            "hard_block": False,
            "features": features.tolist(),
            "keyword_boost_applied": kw_found,
            "keyword_match": kw_match,
            "keyword_boost": round(kw_boost, 4),
        }

    def save(self, path: str = META_MODEL_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

        scaler_path = path.replace("meta_aggregator.pkl", "meta_scaler.pkl")
        if self.scaler is not None:
            joblib.dump(self.scaler, scaler_path)

        logger.info(f"MetaAggregator saved -> {path}")

    @classmethod
    def load(cls, path: str = META_MODEL_PATH) -> "MetaAggregator":
        if not Path(path).exists():
            logger.warning(f"MetaAggregator model not found at {path}. Using fallback weights.")
            return cls()

        # Preferred path: joblib sklearn model + optional scaler
        try:
            model = joblib.load(path)
            scaler_path = path.replace("meta_aggregator.pkl", "meta_scaler.pkl")
            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            logger.info(f"MetaAggregator loaded via joblib -> {path}")
            return cls(model=model, scaler=scaler)
        except Exception as e:
            logger.warning(f"Joblib load failed for {path}: {e}")

        # Backward compatibility: old pickled MetaAggregator object
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            logger.info(f"MetaAggregator loaded via pickle -> {path}")
            return obj
        except Exception as e:
            logger.error(f"Could not load MetaAggregator from {path}: {e}")
            logger.warning("Falling back to default untrained MetaAggregator.")
            return cls()





# ─────────────────────────────────────────────────────────────────────────────
# Pipeline run logger
# ─────────────────────────────────────────────────────────────────────────────

class PipelineLogger:
    """
    Appends every pipeline run to a JSONL log file.
    Each entry contains full evidence vectors for later meta-model retraining.
    """

    def __init__(self, path: str = LOG_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, result: dict) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query_hash": hash(result.get("query_short", "")),
            "action": result["action"],
            "risk_score": result["meta"]["risk_score"],
            "l1_max": result["l1"]["max_score"],
            "l2_prob": result["l2"]["stage1_prob"],
            "l3_cs": result["l3"]["consistency_score"],
            "attack_type": result["l2"]["stage2_label"],
            "features": result["meta"].get("features"),
            "early_exit": bool(result.get("early_exit", False)),
            "blocking_layer": result.get("blocking_layer"),
            "keyword_boost_applied": result["meta"].get("keyword_boost_applied", False),
            "keyword_match": result["meta"].get("keyword_match"),
            "keyword_boost": result["meta"].get("keyword_boost", 0.0),
            "confirmed_attack": None,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def load_confirmed(self) -> tuple:
        X, y = [], []
        if not self.path.exists():
            return np.array(X), y
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row["confirmed_attack"] is not None and row.get("features") is not None:
                    X.append(row["features"])
                    y.append(int(row["confirmed_attack"]))
        return np.array(X), y


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_skipped_l2() -> dict:
    return {
        "stage1_prob": 0.0,
        "stage2_label": None,
        "stage2_conf": 0.0,
        "consistency_score": 0.0,
        "blocked": False,
        "ev": [("Skipped", "Layer 1 early exit")],
    }


def _make_skipped_l3() -> dict:
    return {
        "schema_valid": True,
        "schema_issues": [],
        "boundary_violations": [],
        "consistency_score": 0.0,
        "blocked": False,
        "ev": [("Skipped", "Layer 1 early exit")],
    }


# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline — the single function called by app.py / eval_suite.py
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(document: str,
                 query: str,
                 system_prompt: str,
                 l1_detector=None,
                 l2_classifier=None,
                 l3_monitor=None,
                 meta_aggregator=None,
                 raw_response: Optional[str] = None) -> dict:
    """
    Full pipeline execution.
    """
    from layer1_anomaly import split_chunks

    chunks = split_chunks(document)

    # Layer 1
    if l1_detector is None:
        raise RuntimeError(
            "l1_detector is None. Load with load_detector() and pass to run_pipeline()."
        )
    l1 = l1_detector.scan(chunks)

    # ── Optional early exit after Layer 1 ────────────────────────────────────
    if ENABLE_L1_EARLY_EXIT and l1["blocked"]:
        l2 = _make_skipped_l2()
        l3 = _make_skipped_l3()

        # For early exit, use Layer 1 max score as conservative risk proxy.
        meta = {
            "risk_score": round(float(l1["max_score"]), 4),
            "action": "blocked",
            "confidence": round(abs(float(l1["max_score"]) - 0.5) * 2, 4),
            "hard_block": False,
            "features": None,
            "keyword_boost_applied": False,
            "keyword_match": None,
            "keyword_boost": 0.0,
        }

        result = {
            "chunks": chunks,
            "l1": l1,
            "l2": l2,
            "l3": l3,
            "meta": meta,
            "blocking_layer": "Layer 1 - Anomaly Detection",
            "blocked": True,
            "monitored": False,
            "action": "blocked",
            "timestamp": datetime.utcnow().strftime("%H:%M:%S UTC"),
            "query_short": query[:60] + ("..." if len(query) > 60 else ""),
            "early_exit": True,
        }

        try:
            PipelineLogger().log(result)
        except Exception as e:
            logger.warning(f"Logging failed: {e}")

        logger.info(
            f"action=blocked  risk={meta['risk_score']:.4f}  "
            f"l1={l1['max_score']:.4f}  l2=SKIPPED  l3=SKIPPED  early_exit=True"
        )
        return result

    # Layer 2
    if l2_classifier is None:
        raise RuntimeError(
            "l2_classifier is None. Load with load_classifier() and pass to run_pipeline()."
        )
    l2 = l2_classifier.classify(query, chunks)

    # Layer 3
    if l3_monitor is None:
        raise RuntimeError(
            "l3_monitor is None. Load with load_monitor() and pass to run_pipeline()."
        )
    l3 = l3_monitor.check(query, system_prompt, chunks, l1, l2, raw_response)

    # Meta aggregator
    agg = meta_aggregator or MetaAggregator()
    meta = agg.predict(l1, l2, l3, query=query)

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

    final_blocked = meta["action"] in ("blocked", "hard_block")
    final_monitored = meta["action"] == "monitor"

    result = {
        "chunks": chunks,
        "l1": l1,
        "l2": l2,
        "l3": l3,
        "meta": meta,
        "blocking_layer": blocking_layer,
        "blocked": final_blocked,
        "monitored": final_monitored,
        "action": meta["action"],
        "timestamp": datetime.utcnow().strftime("%H:%M:%S UTC"),
        "query_short": query[:60] + ("..." if len(query) > 60 else ""),
        "early_exit": False,
    }

    try:
        PipelineLogger().log(result)
    except Exception as e:
        logger.warning(f"Logging failed: {e}")

    logger.info(
        f"action={meta['action']}  risk={meta['risk_score']:.4f}  "
        f"l1={l1['max_score']:.4f}  l2={l2['stage1_prob']:.4f}  "
        f"l3={l3['consistency_score']:.4f}  early_exit=False"
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