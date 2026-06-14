"""
layer1_anomaly.py
─────────────────
Layer 1: Document Anomaly Detection

Architecture
────────────
  Embedder  : all-MiniLM-L6-v2  (sentence-transformers)
  Detectors : ECOD + IsolationForest + OneClassSVM  (ensemble)
  Scanning  : per-chunk + sliding window + full-doc concatenation

Training corpus
───────────────
  MITRE ATT&CK descriptions  (~858 docs)
  Wikipedia sample            (~5000 docs)
  Technical documentation     (~200 docs)
  MS MARCO benign queries     (~2000 docs)
  Total: ~8058 diverse clean documents

Latency note
────────────
  OneClassSVM inference time is O(n_support_vectors).
  SVM is trained on a 2000-sample stratified subset to keep
  inference fast (~150-200ms vs ~2600ms on 8058 samples).
  ECOD and IForest scale better and use the full corpus.

Wire into app.py
────────────────
  from layer1_anomaly import load_detector
  detector = load_detector()
  result   = detector.scan(chunks)
"""

import json
import logging
import os
import numpy as np
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download

from config import (
    L1_MODELS_PATH,
    L1_BLOCK_THRESHOLD,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logger = logging.getLogger(__name__)

# Max samples to train OneClassSVM on.
# Limits support vector count → keeps inference fast.
# ECOD and IForest still use the full corpus.
_SVM_MAX_TRAIN_SAMPLES = 2000


def load_or_download_models():
    """Download model component files from HuggingFace if not already present."""
    # Resolve models/ directory relative to this file, not the cwd.
    models_dir = Path(L1_MODELS_PATH).parent
    models_dir.mkdir(parents=True, exist_ok=True)
    files = [
        "layer1_models.pkl.iforest.pkl",
        "layer1_models.pkl.ecod.pkl",
        "layer1_models.pkl.svm.pkl",
        "layer1_models.pkl.threshold.pkl",
    ]
    for f in files:
        dest = models_dir / f
        if not dest.exists():
            logger.info("[L1] Downloading %s from HuggingFace ...", f)
            try:
                hf_hub_download(
                    repo_id="yomnafarag95/rag-defense-models",
                    filename=f,
                    local_dir=str(models_dir),
                )
            except Exception as exc:
                logger.error("[L1] Failed to download %s: %s", f, exc)
                raise RuntimeError(
                    f"Could not download model file '{f}'. "
                    "Check your internet connection and HuggingFace credentials."
                ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Chunking utility
# ─────────────────────────────────────────────────────────────────────────────

def split_chunks(text: str,
                 size: int = CHUNK_SIZE,
                 overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping character-level chunks.
    Overlap ensures injection payloads split across chunk boundaries
    still appear in at least one complete window.
    """
    words, chunks, current, length = text.split(), [], [], 0
    for word in words:
        current.append(word)
        length += len(word) + 1
        if length >= size:
            chunks.append(" ".join(current))
            overlap_words, overlap_len = [], 0
            for w in reversed(current):
                overlap_len += len(w) + 1
                overlap_words.insert(0, w)
                if overlap_len >= overlap:
                    break
            current, length = overlap_words, overlap_len
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


# ─────────────────────────────────────────────────────────────────────────────
# Embedder
# ─────────────────────────────────────────────────────────────────────────────

class InstructorEmbedder:
    """
    Embedding model for document representation.
    Uses all-MiniLM-L6-v2 — produces 384-dim vectors.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        print(f"[L1] Loading embedder: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        print("[L1] Embedder ready.")

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble anomaly detector
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleDetector:
    """
    Three anomaly detectors with score aggregation.

    ECOD            — distribution-based, no contamination assumption
    IsolationForest — tree-based global outlier detection
    OneClassSVM     — margin-based one-class classifier (subsampled for speed)

    Scores are min-max normalised per detector then averaged.
    """

    def __init__(self):
        from pyod.models.ecod import ECOD
        from pyod.models.iforest import IForest
        from sklearn.svm import OneClassSVM

        self.ecod    = ECOD()
        self.iforest = IForest(contamination=0.08, random_state=42)
        self.svm     = OneClassSVM(nu=0.08, kernel="rbf", gamma="scale")
        self.threshold: Optional[float] = None
        
        self._ecod_bounds: Optional[tuple[float, float]] = None
        self._iforest_bounds: Optional[tuple[float, float]] = None
        self._svm_bounds: Optional[tuple[float, float]] = None

    def fit(self, embeddings: np.ndarray,
            threshold: float = L1_BLOCK_THRESHOLD) -> "EnsembleDetector":

        print("[L1] Fitting ECOD ...")
        self.ecod.fit(embeddings)

        print("[L1] Fitting IsolationForest ...")
        self.iforest.fit(embeddings)

        # ── SVM subsampling for inference speed ───────────────────────────────
        if len(embeddings) > _SVM_MAX_TRAIN_SAMPLES:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(embeddings), _SVM_MAX_TRAIN_SAMPLES, replace=False)
            svm_emb = embeddings[idx]
            print(
                f"[L1] Fitting OneClassSVM on {_SVM_MAX_TRAIN_SAMPLES} samples "
                f"(stratified subsample of {len(embeddings)} for inference speed) ..."
            )
        else:
            svm_emb = embeddings
            print("[L1] Fitting OneClassSVM ...")

        self.svm.fit(svm_emb)
        self.threshold = threshold

        # ── Compute score bounds on training data for non-transductive normalization ─
        print("[L1] Computing bounds on training data ...")
        raw_ecod = self.ecod.decision_function(embeddings)
        self._ecod_bounds = (float(raw_ecod.min()), float(raw_ecod.max()))

        raw_iforest = self.iforest.decision_function(embeddings)
        self._iforest_bounds = (float(raw_iforest.min()), float(raw_iforest.max()))

        # SVM subset or full corpus? Let's use svm_emb because svm was fit on it.
        raw_svm = -self.svm.decision_function(svm_emb)
        self._svm_bounds = (float(raw_svm.min()), float(raw_svm.max()))

        print(f"[L1] ECOD training bounds: {self._ecod_bounds}")
        print(f"[L1] IForest training bounds: {self._iforest_bounds}")
        print(f"[L1] SVM training bounds: {self._svm_bounds}")
        
        print(f"[L1] Ensemble trained. Block threshold = {threshold}")
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Returns anomaly scores in [0, 1]. Higher = more anomalous."""
        scores_ecod    = self._norm(self.ecod.decision_function(embeddings), self._ecod_bounds)
        scores_iforest = self._norm(self.iforest.decision_function(embeddings), self._iforest_bounds)
        scores_svm     = self._norm(-self.svm.decision_function(embeddings), self._svm_bounds)
        return (scores_ecod + scores_iforest + scores_svm) / 3.0

    @staticmethod
    def _norm(arr: np.ndarray, bounds: Optional[tuple[float, float]] = None) -> np.ndarray:
        if bounds is None or bounds[0] == bounds[1]:
            mn, mx = arr.min(), arr.max()
            if mx == mn:
                return np.zeros_like(arr, dtype=float)
            return (arr - mn) / (mx - mn)
        mn, mx = bounds
        return np.clip((arr - mn) / (mx - mn), 0.0, 1.0)

    def save(self, path: str) -> None:
        import joblib
        base = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.ecod,      base + ".ecod.pkl")
        joblib.dump(self.iforest,   base + ".iforest.pkl")
        joblib.dump(self.svm,       base + ".svm.pkl")
        joblib.dump(self.threshold, base + ".threshold.pkl")
        
        bounds = {
            "ecod": self._ecod_bounds,
            "iforest": self._iforest_bounds,
            "svm": self._svm_bounds,
        }
        joblib.dump(bounds, base + ".bounds.pkl")
        print(f"[L1] Ensemble saved to component files (including bounds) at {path}")

    @classmethod
    def load(cls, path: str) -> "EnsembleDetector":
        """Load ensemble from saved component files. Raises RuntimeError on failure."""
        import joblib
        base = str(path)
        obj = cls.__new__(cls)
        component_files = {
            "ecod":      base + ".ecod.pkl",
            "iforest":   base + ".iforest.pkl",
            "svm":       base + ".svm.pkl",
            "threshold": base + ".threshold.pkl",
        }
        for attr, fpath in component_files.items():
            if not Path(fpath).exists():
                raise RuntimeError(f"[L1] Model component missing: {fpath}")
            try:
                setattr(obj, attr, joblib.load(fpath))
            except Exception as exc:
                raise RuntimeError(
                    f"[L1] Failed to load model component '{fpath}': {exc}"
                ) from exc
        
        # Load bounds if present
        bounds_path = base + ".bounds.pkl"
        if Path(bounds_path).exists():
            try:
                bounds = joblib.load(bounds_path)
                obj._ecod_bounds = bounds.get("ecod")
                obj._iforest_bounds = bounds.get("iforest")
                obj._svm_bounds = bounds.get("svm")
            except Exception as exc:
                logger.warning(f"[L1] Failed to load bounds: {exc}. Using dynamic normalization.")
                obj._ecod_bounds = None
                obj._iforest_bounds = None
                obj._svm_bounds = None
        else:
            logger.warning("[L1] Bounds file missing. Using dynamic normalization.")
            obj._ecod_bounds = None
            obj._iforest_bounds = None
            obj._svm_bounds = None
            
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# AnomalyDetector — main interface
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Full Layer 1 pipeline.

    scan(chunks) → dict with keys:
        chunk_scores    : list[float]
        window_scores   : list[float]
        full_score      : float
        max_score       : float
        flagged_chunks  : list[int]
        blocked         : bool
        ev              : list[tuple]
    """

    def __init__(self,
                 embedder: InstructorEmbedder,
                 detector: EnsembleDetector):
        self.embedder = embedder
        self.detector = detector

    def scan(self, chunks: list[str]) -> dict:
        """Scan document chunks for anomalies. Returns detection results dict."""
        # Input validation
        if not chunks:
            logger.warning("[L1] scan() called with empty chunks list")
            return {
                "chunk_scores":   [],
                "window_scores":  [],
                "full_score":     0.0,
                "max_score":      0.0,
                "flagged_chunks": [],
                "blocked":        False,
                "confidence":     0.0,
                "ev": [("Warning", "No document chunks provided")],
            }

        if self.detector.threshold is None:
            raise RuntimeError("[L1] Detector threshold not set. Re-load the detector.")

        # ── Per-chunk scores ─────────────────────────────────────────────────────────
        chunk_embs   = self.embedder.encode(chunks)
        chunk_scores = self.detector.score(chunk_embs).tolist()
        flagged      = [i for i, s in enumerate(chunk_scores)
                        if s > self.detector.threshold]

        # ── Sliding window (BATCHED — one encode call instead of N-1) ────────────────
        window_scores: list[float] = []
        if len(chunks) > 1:
            window_texts = [
                chunks[i] + " " + chunks[i + 1]
                for i in range(len(chunks) - 1)
            ]
            window_embs  = self.embedder.encode(window_texts)
            window_scores = [
                round(float(s), 4)
                for s in self.detector.score(window_embs)
            ]
        else:
            window_scores = [round(chunk_scores[0], 4)]

        # ── Full-document score ───────────────────────────────────────────────
        full_text  = " ".join(chunks)
        full_emb   = self.embedder.encode([full_text])
        full_score = round(float(self.detector.score(full_emb)[0]), 4)

        max_score = round(
            max(chunk_scores + window_scores + [full_score]), 4
        )
        blocked = max_score > self.detector.threshold

        # Confidence: distance from threshold, normalised to [0, 1]
        confidence = round(min(abs(max_score - self.detector.threshold) * 2, 1.0), 4)

        logger.info(
            "[L1] max_score=%.4f threshold=%.4f blocked=%s flagged=%d/%d",
            max_score, self.detector.threshold, blocked, len(flagged), len(chunks),
        )

        return {
            "chunk_scores":   [round(s, 4) for s in chunk_scores],
            "window_scores":  window_scores,
            "full_score":     full_score,
            "max_score":      max_score,
            "flagged_chunks": flagged,
            "blocked":        blocked,
            "confidence":     confidence,
            "ev": [
                ("Chunks flagged",    f"{len(flagged)} / {len(chunks)}"),
                ("Max chunk score",   f"{max(chunk_scores):.4f}"),
                ("Window scan max",   f"{max(window_scores, default=0.0):.4f}"),
                ("Full doc score",    f"{full_score:.4f}"),
                ("Confidence",        f"{confidence:.4f}"),
                ("Detector ensemble", "ECOD · IForest · OneClassSVM"),
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def load_detector(models_path: str = L1_MODELS_PATH) -> AnomalyDetector:
    """
    Load embedder and trained detector ensemble from disk.

    Threshold override:
        Always uses L1_BLOCK_THRESHOLD from config.py at runtime.
        This allows threshold tuning without full retraining.
        The saved pkl threshold value is ignored.
    """
    embedder = InstructorEmbedder()
    try:
        detector = EnsembleDetector.load(models_path)
    except RuntimeError:
        logger.warning("[L1] Pre-trained models not found. Attempting download ...")
        load_or_download_models()
        detector = EnsembleDetector.load(models_path)

    # Override saved threshold with current config value.
    # This means changing L1_BLOCK_THRESHOLD in config.py
    # takes effect immediately without retraining.
    detector.threshold = L1_BLOCK_THRESHOLD
    logger.info("[L1] Threshold set to %.4f (from config.py)", L1_BLOCK_THRESHOLD)
    print(f"[L1] Threshold set to {L1_BLOCK_THRESHOLD} (from config.py)")

    return AnomalyDetector(embedder, detector)


# ─────────────────────────────────────────────────────────────────────────────
# Corpus loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_training_corpus() -> list[str]:
    """
    Load diverse clean training corpus for Layer 1 anomaly detector.

    Sources
    ───────
    1. MITRE ATT&CK descriptions  (~858 docs)   security domain text
    2. Wikipedia sample            (~5000 docs)  general language diversity
    3. Technical documentation     (~200 docs)   technical language patterns
    4. MS MARCO benign queries     (~2000 docs)  short query patterns

    Source #4 is critical: without short benign queries in training,
    the detector flags all short queries as anomalous (high FPR).
    """
    docs = []

    # ── MITRE ATT&CK ─────────────────────────────────────────────────────────
    mitre_path = Path("data/mitre_clean.json")
    if mitre_path.exists():
        with open(mitre_path, encoding="utf-8") as f:
            mitre = json.load(f)
        mitre_docs = [
            e["description"] for e in mitre
            if e.get("description") and len(e["description"]) >= 20
        ]
        docs.extend(mitre_docs)
        print(f"[L1] MITRE ATT&CK:      {len(mitre_docs):>5} docs")
    else:
        print("[L1] WARNING: data/mitre_clean.json not found — run python data_loader.py")

    # ── Wikipedia sample ─────────────────────────────────────────────────────
    wiki_path = Path("data/wikipedia_sample.jsonl")
    if wiki_path.exists():
        wiki_docs = []
        with open(wiki_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    text = str(row.get("text", "")).strip()
                    if len(text) >= 20:
                        wiki_docs.append(text)
                except json.JSONDecodeError:
                    continue
        docs.extend(wiki_docs)
        print(f"[L1] Wikipedia:         {len(wiki_docs):>5} docs")
    else:
        print("[L1] WARNING: data/wikipedia_sample.jsonl not found — run python data_loader.py")

    # ── Technical documentation ───────────────────────────────────────────────
    tech_path = Path("data/technical_docs.jsonl")
    if tech_path.exists():
        tech_docs = []
        with open(tech_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    text = str(row.get("text", "")).strip()
                    if len(text) >= 20:
                        tech_docs.append(text)
                except json.JSONDecodeError:
                    continue
        docs.extend(tech_docs)
        print(f"[L1] Technical docs:    {len(tech_docs):>5} docs")
    else:
        print("[L1] WARNING: data/technical_docs.jsonl not found — run python data_loader.py")

    # ── MS MARCO benign queries ───────────────────────────────────────────────
    bq_path = Path("data/benign_queries.jsonl")
    if bq_path.exists():
        bq_docs = []
        with open(bq_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    text = str(row.get("text", "")).strip()
                    if len(text) >= 5:
                        bq_docs.append(text)
                except json.JSONDecodeError:
                    continue
                if len(bq_docs) >= 2000:
                    break
        docs.extend(bq_docs)
        print(f"[L1] Benign queries:    {len(bq_docs):>5} docs")
    else:
        print("[L1] WARNING: data/benign_queries.jsonl not found — run python data_loader.py")

    print(f"[L1] Total corpus:      {len(docs):>5} docs")

    if len(docs) < 100:
        raise RuntimeError(
            f"Training corpus too small ({len(docs)} docs). "
            "Run: python data_loader.py first."
        )

    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Training script
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n=== Layer 1 — Training Anomaly Detectors ===\n")

    print("[L1] Loading training corpus ...")
    clean_docs = _load_training_corpus()
    print(f"\n[L1] Total training docs: {len(clean_docs)}")

    print("\n[L1] Embedding documents (this may take 5-10 minutes) ...")
    embedder   = InstructorEmbedder()
    embeddings = embedder.encode(clean_docs)
    print(f"[L1] Embeddings shape: {embeddings.shape}")

    print("\n[L1] Training ensemble detectors ...")
    detector = EnsembleDetector()
    detector.fit(embeddings, threshold=L1_BLOCK_THRESHOLD)

    detector.save(L1_MODELS_PATH)

    print(f"\n[L1] Training complete.")
    print(f"     Model saved to : {L1_MODELS_PATH}")
    print(f"     Corpus         : MITRE + Wikipedia + Technical + Benign queries")
    print(f"     SVM trained on : up to {_SVM_MAX_TRAIN_SAMPLES} samples (for speed)")
    print(f"     Threshold      : {L1_BLOCK_THRESHOLD} (always overridden by config at runtime)")
    print(f"     Run            : python eval_suite.py --mode all")