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

from huggingface_hub import hf_hub_download
import os


def load_or_download_models():
    if not os.path.exists("models/"):
        os.makedirs("models/")
    files = [
        "layer1_models.pkl.iforest.pkl",
        "layer1_models.pkl.ecod.pkl",
        "layer1_models.pkl.svm.pkl",
        "layer1_models.pkl.threshold.pkl",
    ]
    for f in files:
        if not os.path.exists(f"models/{f}"):
            hf_hub_download(
                repo_id="yomnafarag95/rag-defense-models",
                filename=f,
                local_dir="models/",
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

# Max samples to train OneClassSVM on.
# Limits support vector count → keeps inference fast.
# ECOD and IForest still use the full corpus.
_SVM_MAX_TRAIN_SAMPLES = 2000


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

    def fit(self, embeddings: np.ndarray,
            threshold: float = L1_BLOCK_THRESHOLD) -> "EnsembleDetector":

        print("[L1] Fitting ECOD ...")
        self.ecod.fit(embeddings)

        print("[L1] Fitting IsolationForest ...")
        self.iforest.fit(embeddings)

        # ── SVM subsampling for inference speed ───────────────────────────────
        # OneClassSVM scoring is O(n_support_vectors).
        # nu=0.08 on 8058 samples → ~645 support vectors → ~2600ms per query.
        # Subsampling to 2000 → ~160 support vectors → ~200ms per query.
        # ECOD and IForest are not subsampled — they scale well.
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
        print(f"[L1] Ensemble trained. Block threshold = {threshold}")
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Returns anomaly scores in [0, 1]. Higher = more anomalous."""
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
        import joblib
        base = str(path)
        obj = cls.__new__(cls)
        obj.ecod      = joblib.load(base + ".ecod.pkl")
        obj.iforest   = joblib.load(base + ".iforest.pkl")
        obj.svm       = joblib.load(base + ".svm.pkl")
        obj.threshold = joblib.load(base + ".threshold.pkl")
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

        # ── Per-chunk scores ─────────────────────────────────────────────────
        chunk_embs   = self.embedder.encode(chunks)
        chunk_scores = self.detector.score(chunk_embs).tolist()
        flagged      = [i for i, s in enumerate(chunk_scores)
                        if s > self.detector.threshold]

        # ── Sliding window ───────────────────────────────────────────────────
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
    detector = EnsembleDetector.load(models_path)

    # Override saved threshold with current config value.
    # This means changing L1_BLOCK_THRESHOLD in config.py
    # takes effect immediately without retraining.
    detector.threshold = L1_BLOCK_THRESHOLD
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