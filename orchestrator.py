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
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

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
    CANARY_TOKEN,
    CANARY_DETECTION_ENABLED,
    STATEFUL_HISTORY_LIMIT,
    STATEFUL_DRIFT_THRESHOLD,
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
        # l2_consistency_score features scale naturally since training query-document pairs are matched.
        l2_consist_safe = float(l2["consistency_score"])
        return np.array([
            l1["max_score"],
            max(l1["window_scores"], default=0.0),
            l1["full_score"],
            l2["stage1_prob"],
            l2_consist_safe,
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
            raise RuntimeError(
                "Meta-aggregator model file is missing or not trained. "
                "Please run training first: python train_meta_aggregator.py"
            )

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

        # Build scaler path alongside the model file (robust to path changes)
        scaler_path = str(Path(path).parent / "meta_scaler.pkl")
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
            scaler_path = str(Path(path).parent / "meta_scaler.pkl")
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
            raise RuntimeError(
                f"Could not load MetaAggregator from {path}: {e}. "
                "Please run: python train_meta_aggregator.py"
            )





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
        import hashlib
        # Use stable sha256 hash (Python's built-in hash() is not deterministic
        # across interpreter restarts since Python 3.3 hash randomisation)
        query_short = result.get("query_short", "")
        query_hash = hashlib.sha256(query_short.encode("utf-8")).hexdigest()[:8]
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query_hash": query_hash,
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
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as exc:
            logger.error("[PipelineLogger] Failed to write log entry: %s", exc)

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

class StatefulAttackTracker:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StatefulAttackTracker, cls).__new__(cls)
                cls._instance.history = defaultdict(list)
                cls._instance.lock = threading.Lock()
            return cls._instance

    def add_score(self, session_id: str, score: float):
        if not session_id:
            return
        with self.lock:
            hist = self.history[session_id]
            hist.append(score)
            if len(hist) > STATEFUL_HISTORY_LIMIT:
                hist.pop(0)

    def get_drift_score(self, session_id: str) -> float:
        if not session_id:
            return 0.0
        with self.lock:
            hist = self.history[session_id]
            if not hist:
                return 0.0
            weights = [0.5 ** i for i in range(len(hist))][::-1]
            total_weight = sum(weights)
            weighted_score = sum(h * w for h, w in zip(hist, weights)) / total_weight
            return round(weighted_score, 4)

    def get_session_history(self, session_id: str) -> list[float]:
        """Return the raw score history for a session (for UI display)."""
        if not session_id:
            return []
        with self.lock:
            return list(self.history.get(session_id, []))

    def get_trend(self, session_id: str) -> str:
        """Return 'rising', 'falling', or 'stable' based on recent score trend."""
        hist = self.get_session_history(session_id)
        if len(hist) < 2:
            return "stable"
        delta = hist[-1] - hist[-2]
        if delta > 0.05:
            return "rising"
        elif delta < -0.05:
            return "falling"
        return "stable"


def _make_canary_block_result(source: str, query: str) -> dict:
    meta = {
        "risk_score": 1.0,
        "action": "hard_block",
        "confidence": 1.0,
        "hard_block": True,
        "features": [1.0] * 10,
        "keyword_boost_applied": False,
        "keyword_match": f"canary_token_leak:{source}",
        "keyword_boost": 0.0,
    }
    return {
        "chunks": [],
        "l1": {"blocked": True, "max_score": 1.0, "window_scores": [], "full_score": 1.0, "flagged_chunks": [], "ev": [("Canary Check", f"Failed: detected in {source}")]},
        "l2": _make_skipped_l2(),
        "l3": _make_skipped_l3(),
        "meta": meta,
        "blocking_layer": "Canary Honeypot Detector",
        "blocked": True,
        "monitored": False,
        "action": "hard_block",
        "timestamp": datetime.utcnow().strftime("%H:%M:%S UTC"),
        "query_short": query[:60] + ("..." if len(query) > 60 else ""),
        "early_exit": True,
    }


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
                 raw_response: Optional[str] = None,
                 session_id: Optional[str] = None,
                 canary_manager=None) -> dict:
    """
    Full pipeline execution.

    Raises ValueError for empty inputs.
    Raises RuntimeError if detectors are not provided.
    """
    import unicodedata
    def sanitize_text(text: str) -> str:
        if not text:
            return ""
        text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C' or c in '\n\t')
        return unicodedata.normalize('NFKC', text)

    document = sanitize_text(document)
    query = sanitize_text(query)
    system_prompt = sanitize_text(system_prompt)
    if raw_response is not None:
        raw_response = sanitize_text(raw_response)

    from layer1_anomaly import split_chunks

    # Input validation
    if not document or not document.strip():
        raise ValueError("run_pipeline: 'document' must not be empty.")
    if not query or not query.strip():
        raise ValueError("run_pipeline: 'query' must not be empty.")
    if not system_prompt:
        system_prompt = "Answer using only the provided knowledge base."

    # Active Defense: Canary token checks on user query
    if CANARY_DETECTION_ENABLED and CANARY_TOKEN and CANARY_TOKEN.lower() in query.lower():
        logger.warning(f"[Canary] Canary token '{CANARY_TOKEN}' detected in user query.")
        return _make_canary_block_result("query", query)

    # Active Defense: Canary token checks on raw response (post-response check)
    if CANARY_DETECTION_ENABLED and CANARY_TOKEN and raw_response and CANARY_TOKEN.lower() in raw_response.lower():
        logger.warning(f"[Canary] Canary token '{CANARY_TOKEN}' detected in response.")
        return _make_canary_block_result("response", query)

    chunks = split_chunks(document)

    # Active Defense: Canary context check on retrieved document chunks
    if canary_manager is not None and CANARY_DETECTION_ENABLED:
        context_leaked, leaked_chunk = canary_manager.check_context(chunks)
        if context_leaked:
            logger.warning("[Canary] Canary token detected in document context. Hard-blocking.")
            result = _make_canary_block_result("document_context", query)
            result["l1"]["ev"].append(("Canary Context", f"Token found in retrieved document chunk: {(leaked_chunk or '')[:60]}"))
            return result

    # Layer 1 & 2 Execution in Parallel
    if l1_detector is None:
        raise RuntimeError(
            "l1_detector is None. Load with load_detector() and pass to run_pipeline()."
        )
    if l2_classifier is None:
        raise RuntimeError(
            "l2_classifier is None. Load with load_classifier() and pass to run_pipeline()."
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_l1 = executor.submit(l1_detector.scan, chunks)
        future_l2 = executor.submit(l2_classifier.classify, query, chunks)
        try:
            l1 = future_l1.result()
        except Exception as exc:
            logger.error("Layer 1 scanning failed: %s", exc)
            raise RuntimeError(f"Layer 1 scan failure: {exc}") from exc
        try:
            l2 = future_l2.result()
        except Exception as exc:
            logger.error("Layer 2 classification failed: %s", exc)
            raise RuntimeError(f"Layer 2 classification failure: {exc}") from exc

    # ── Optional early exit after Layer 1 ────────────────────────────────────
    if ENABLE_L1_EARLY_EXIT and l1["blocked"]:
        l3 = _make_skipped_l3()

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

        # Stateful history update for early exit
        if session_id:
            StatefulAttackTracker().add_score(session_id, meta["risk_score"])

        try:
            PipelineLogger().log(result)
        except Exception as e:
            logger.warning(f"Logging failed: {e}")

        logger.info(
            f"action=blocked  risk={meta['risk_score']:.4f}  "
            f"l1={l1['max_score']:.4f}  l2={l2['stage1_prob']:.4f}  l3=SKIPPED  early_exit=True"
        )
        return result

    # Layer 3
    if l3_monitor is None:
        raise RuntimeError(
            "l3_monitor is None. Load with load_monitor() and pass to run_pipeline()."
        )
    l3 = l3_monitor.check(query, system_prompt, chunks, l1, l2, raw_response)

    # Meta aggregator
    if meta_aggregator is None:
        logger.warning(
            "[pipeline] meta_aggregator not provided — using untrained fallback weights. "
            "Run 'python orchestrator.py retrain' after labelling log entries."
        )
        agg = MetaAggregator()
    else:
        agg = meta_aggregator
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

    # Stateful multi-turn Crescendo protection
    if session_id:
        tracker = StatefulAttackTracker()
        tracker.add_score(session_id, meta["risk_score"])
        drift = tracker.get_drift_score(session_id)
        if drift > STATEFUL_DRIFT_THRESHOLD and meta["action"] not in ("blocked", "hard_block"):
            logger.warning(f"[Stateful] Multi-turn drift score {drift} exceeded threshold {STATEFUL_DRIFT_THRESHOLD}. Escalating to blocked.")
            meta["action"] = "blocked"
            meta["risk_score"] = max(meta["risk_score"], drift)
            result["blocked"] = True
            result["action"] = "blocked"
            result["blocking_layer"] = "Stateful Multi-Turn Tracker"
            result["meta"]["risk_score"] = meta["risk_score"]
            result["meta"]["action"] = "blocked"

    try:
        PipelineLogger().log(result)
    except Exception as e:
        logger.warning(f"Logging failed: {e}")

    logger.info(
        f"action={result['action']}  risk={result['meta']['risk_score']:.4f}  "
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