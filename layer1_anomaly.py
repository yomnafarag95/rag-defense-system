"""
layer1_anomaly.py
─────────────────
Layer 1: Document Anomaly Detection

Architecture
────────────
  Embedder  : all-MiniLM-L6-v2  (sentence-transformers, fully compatible)
  Detectors : ECOD + IsolationForest + OneClassSVM  (ensemble)
  Scanning  : per-chunk + sliding window + full-doc concatenation

Wire into app.py
────────────────
  from layer1_anomaly import AnomalyDetector
  detector = load_detector()          # wrap in @st.cache_resource
  result   = detector.scan(chunks)
"""
from huggingface_hub import hf_hub_download
import os

def load_or_download_models():
    if not os.path.exists("models/"):
        os.makedirs("models/")
    
    files = [
        "layer1_models.pkl.iforest.pkl",
        "layer1_models.pkl.ecod.pkl", 
        "layer1_models.pkl.svm.pkl",
        "layer1_models.pkl.threshold.pkl"
    ]
    
    for f in files:
        if not os.path.exists(f"models/{f}"):
            hf_hub_download(
                repo_id="yomnafarag95/rag-defense-models",
                filename=f,
                local_dir="models/"
            )
            
import json
import numpy as np
from pathlib import Path
from typing import Optional

from config import (
    L1_MODELS_PATH,
    L1_BLOCK_THRESHOLD,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# ─────────────────────────────────────────────────────────────────────────────
# Chunking utility (shared across all layers)
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
            overlap_words = []
            overlap_len   = 0
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
# Embedder — uses all-MiniLM-L6-v2 (already installed, fully compatible)
# ─────────────────────────────────────────────────────────────────────────────

class InstructorEmbedder:
    """
    Embedding model for document representation.
    Uses all-MiniLM-L6-v2 from sentence-transformers.
    Produces 384-dimensional vectors for each input text.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        print(f"[L1] Loading embedder: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        print(f"[L1] Embedder ready.")

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

    ECOD            — distribution-based, no contamination assumption needed
    IsolationForest — tree-based global outlier detection
    OneClassSVM     — margin-based one-class classifier

    Scores are min-max normalised per detector then averaged.
    Threshold is set via L1_BLOCK_THRESHOLD in config.py (default 0.50).
    """

    def __init__(self):
        from pyod.models.ecod import ECOD
        from pyod.models.iforest import IForest
        from sklearn.svm import OneClassSVM

        self.ecod    = ECOD()
        self.iforest = IForest(contamination=0.08, random_state=42)
        self.svm     = OneClassSVM(nu=0.08, kernel="rbf", gamma="scale")
        self.threshold: Optional[float] = None

    def fit(self, embeddings: np.ndarray,
            threshold: float = L1_BLOCK_THRESHOLD) -> "EnsembleDetector":
        print("[L1] Fitting ECOD ...")
        self.ecod.fit(embeddings)

        print("[L1] Fitting IsolationForest ...")
        self.iforest.fit(embeddings)

        print("[L1] Fitting OneClassSVM ...")
        self.svm.fit(embeddings)

        self.threshold = threshold
        print(f"[L1] Ensemble trained. Block threshold = {threshold}")
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Returns anomaly scores in [0, 1] for each embedding.
        Higher = more anomalous.
        """
        scores_ecod    = self._norm(self.ecod.decision_function(embeddings))
        scores_iforest = self._norm(self.iforest.decision_function(embeddings))
        scores_svm     = self._norm(-self.svm.decision_function(embeddings))
        return (scores_ecod + scores_iforest + scores_svm) / 3.0

    @staticmethod
    def _norm(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.zeros_like(arr, dtype=float)
        return (arr - mn) / (mx - mn)

    def save(self, path: str) -> None:
        """
        Save each detector component separately.
        Avoids the __main__ vs module name mismatch entirely.
        """
        import joblib
        base = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.ecod,      base + ".ecod.pkl")
        joblib.dump(self.iforest,   base + ".iforest.pkl")
        joblib.dump(self.svm,       base + ".svm.pkl")
        joblib.dump(self.threshold, base + ".threshold.pkl")
        print(f"[L1] Ensemble saved to 4 component files at {path}")

    @classmethod
    def load(cls, path: str) -> "EnsembleDetector":
        """
        Reconstruct EnsembleDetector by loading each sklearn/pyod component
        separately. No class name lookup — works from any calling module.
        """
        import joblib
        base = str(path)
        obj = cls.__new__(cls)
        obj.ecod      = joblib.load(base + ".ecod.pkl")
        obj.iforest   = joblib.load(base + ".iforest.pkl")
        obj.svm       = joblib.load(base + ".svm.pkl")
        obj.threshold = joblib.load(base + ".threshold.pkl")
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# AnomalyDetector — main interface called by app.py
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Full Layer 1 pipeline.

    scan(chunks) → dict with keys:
        chunk_scores    : list[float]  — per-chunk anomaly score
        window_scores   : list[float]  — sliding-window (pairs) scores
        full_score      : float        — whole-document score
        max_score       : float        — worst signal across all scans
        flagged_chunks  : list[int]    — indices of chunks above threshold
        blocked         : bool
        ev              : list[tuple]  — evidence rows for UI table
    """

    def __init__(self,
                 embedder: InstructorEmbedder,
                 detector: EnsembleDetector):
        self.embedder = embedder
        self.detector = detector

    def scan(self, chunks: list[str]) -> dict:

        # ── Per-chunk scores ──────────────────────────────────────────────────
        chunk_embs   = self.embedder.encode(chunks)
        chunk_scores = self.detector.score(chunk_embs).tolist()
        flagged      = [i for i, s in enumerate(chunk_scores)
                        if s > self.detector.threshold]

        # ── Sliding window — catches payload splitting across chunks ──────────
        window_scores = []
        for i in range(len(chunks) - 1):
            combined = chunks[i] + " " + chunks[i + 1]
            emb      = self.embedder.encode([combined])
            score    = float(self.detector.score(emb)[0])
            window_scores.append(round(score, 4))

        # ── Full-document score ───────────────────────────────────────────────
        full_text  = " ".join(chunks)
        full_emb   = self.embedder.encode([full_text])
        full_score = round(float(self.detector.score(full_emb)[0]), 4)

        max_score = round(
            max(chunk_scores + window_scores + [full_score]), 4
        )
        blocked = max_score > self.detector.threshold

        return {
            "chunk_scores":   [round(s, 4) for s in chunk_scores],
            "window_scores":  window_scores,
            "full_score":     full_score,
            "max_score":      max_score,
            "flagged_chunks": flagged,
            "blocked":        blocked,
            "ev": [
                ("Chunks flagged",    f"{len(flagged)} / {len(chunks)}"),
                ("Max chunk score",   f"{max(chunk_scores):.4f}"),
                ("Window scan max",   f"{max(window_scores, default=0.0):.4f}"),
                ("Full doc score",    f"{full_score:.4f}"),
                ("Detector ensemble", "ECOD · IForest · OneClassSVM"),
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory (use this in app.py with @st.cache_resource)
# ─────────────────────────────────────────────────────────────────────────────

def load_detector(models_path: str = L1_MODELS_PATH) -> AnomalyDetector:
    """
    Load embedder and trained detector ensemble from disk.

    Usage in app.py:
        import streamlit as st
        from layer1_anomaly import load_detector

        @st.cache_resource
        def get_l1():
            return load_detector()
    """
    embedder = InstructorEmbedder()
    detector = EnsembleDetector.load(models_path)
    return AnomalyDetector(embedder, detector)


# ─────────────────────────────────────────────────────────────────────────────
# Training script  (python layer1_anomaly.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n=== Layer 1 — Training Anomaly Detectors ===\n")

    # 1. Load clean documents from MITRE
    mitre_path = Path("data/mitre_clean.json")
    if not mitre_path.exists():
        raise FileNotFoundError(
            "data/mitre_clean.json not found.\n"
            "Run: python data_loader.py first."
        )

    with open(mitre_path) as f:
        mitre = json.load(f)

    clean_docs = [e["description"] for e in mitre if e.get("description")]
    print(f"[L1] Loaded {len(clean_docs)} clean MITRE documents.")

    # 2. Embed
    print("[L1] Embedding documents (this takes 1-3 minutes) ...")
    embedder   = InstructorEmbedder()
    embeddings = embedder.encode(clean_docs)
    print(f"[L1] Embeddings shape: {embeddings.shape}")

    # 3. Train ensemble
    print("[L1] Training ensemble detectors ...")
    detector = EnsembleDetector()
    detector.fit(embeddings, threshold=L1_BLOCK_THRESHOLD)

    # 4. Save with joblib (avoids pickle __main__ module bug)
    detector.save(L1_MODELS_PATH)

    print(f"\n[L1] Training complete.")
    print(f"     Model saved to: {L1_MODELS_PATH}")
    print(f"     Run: streamlit run app.py")