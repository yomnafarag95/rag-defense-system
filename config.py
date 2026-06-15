"""
config.py
─────────
Single source of truth for every tunable constant in the pipeline.
Change values here; nothing else needs editing.
"""

from pathlib import Path
import os

# ── Project root (directory that contains this config.py) ─────────────────────
_ROOT = Path(__file__).parent

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE         = 200
CHUNK_OVERLAP      = 40

# ── Layer 1 — Anomaly Detection ───────────────────────────────────────────────
L1_EMBEDDER_MODEL    = "all-MiniLM-L6-v2"
L1_MODELS_PATH       = str(_ROOT / "models" / "layer1_models.pkl")
L1_BLOCK_THRESHOLD   = 0.85  # Raised to 0.85 to adapt to non-transductive fixed-bounds normalization.
                               # load_detector() always overrides saved pkl value
                               # with this constant so retraining is not required
                               # for threshold-only changes.
ENABLE_L1_EARLY_EXIT = True

# ── Layer 2 — Intent Classifier ───────────────────────────────────────────────
L2_BASE_MODEL              = "microsoft/deberta-v3-small"  # Smaller/faster base for fine-tune
L2_FINETUNED_PATH          = str(_ROOT / "models" / "layer2_finetuned")  # Output of fine_tune_l2.py
L2_USE_FINETUNED           = False  # Use fine-tuned model if available; falls back to pretrained
L2_STAGE1_THRESHOLD        = 0.60    # Query-level block threshold
L2_DOC_SCAN_CHUNKS         = None    # Max document chunks to scan (None means scan all)
L2_DOC_PATTERN_THRESHOLD   = 0.60    # Document pattern score to trigger block
L2_NUM_ATTACK_TYPES        = 6
ENABLE_L2_EARLY_EXIT       = True
SEMANTIC_CACHE_THRESHOLD   = 0.98

ATTACK_LABELS = {
    0: "instruction_override",
    1: "role_manipulation",
    2: "payload_splitting",
    3: "indirect_injection",
    4: "encoding_obfuscation",
    5: "context_exhaustion",
}

ATTACK_DISPLAY = {
    "instruction_override":  "Instruction Override",
    "role_manipulation":     "Role Manipulation",
    "payload_splitting":     "Payload Splitting",
    "indirect_injection":    "Indirect Injection",
    "encoding_obfuscation":  "Encoding Obfuscation",
    "context_exhaustion":    "Context Exhaustion",
}

# ── Layer 3 — Behavioral Monitor ──────────────────────────────────────────────
L3_CONSISTENCY_MODEL      = "cross-encoder/ms-marco-MiniLM-L-12-v2"
L3_FINETUNED_PATH         = str(_ROOT / "models" / "layer3_consistency")
L3_CONSISTENCY_THRESHOLD  = 0.55
L3_MAX_QUERY_LEN          = 500
L3_MAX_DOC_LEN            = 3000

SENSITIVE_PATTERNS = [
    (r"[\w.]+@[\w.]+\.\w+",                               "email"),
    (r"\b[A-Za-z0-9]{40,}\b",                             "api_key"),
    (r"\b(?:password|passwd|secret|token)\s*[:=]\s*\S+",  "credential"),
]

# ── Meta Aggregator ───────────────────────────────────────────────────────────
META_MODEL_PATH        = str(_ROOT / "models" / "meta_aggregator.pkl")
META_HARD_BLOCK_SINGLE = 1.01
META_HARD_BLOCK_VIOLS  = 2
META_BLOCK_THRESHOLD   = 0.35   # Optimal operating point (Run 1): benign scores cluster 0.12–0.30.
                                # Threshold 0.35 keeps FPR_prevention=0% while blocking hard attacks.
META_MONITOR_THRESHOLD = 0.31   # Optimal detection threshold: only 1/423 benign scored above 0.30
                                # (FPR_detection=0.24%). Attacks scoring 0.30–0.35 are flagged as
                                # "monitored". Below 0.30 is the unresolvable overlap zone requiring
                                # a stronger meta-model to penetrate further.

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_PATH          = str(_ROOT / "logs" / "pipeline.jsonl")
MAX_HISTORY_ITEMS = 20

# ── Canary & Stateful Detection ───────────────────────────────────────────────
# SECURITY: CANARY_TOKEN must be set via environment variable, never hardcoded.
# Generate a strong random token and add it to your .env file:
#   python -c "import secrets; print('RAG_CANARY_TOKEN=' + secrets.token_hex(16))"
# The empty-string default disables canary checking if the env var is not set.
CANARY_TOKEN               = os.environ.get("RAG_CANARY_TOKEN", "")
CANARY_INJECT_COUNT        = 3     # Number of honeypot docs to inject per vector store
CANARY_DETECTION_ENABLED   = True  # Set False to disable canary context/response checks

STATEFUL_HISTORY_LIMIT   = 5
# Lowered from 0.85 → 0.72 to detect gradual Crescendo-style attacks earlier.
# At 0.85, a 3-turn attack with scores [0.40, 0.60, 0.80] would score
# weighted 0.62 — below threshold. At 0.72, it triggers a block.
STATEFUL_DRIFT_THRESHOLD = 0.72

# ── Device ────────────────────────────────────────────────────────────────────
import torch as _torch
DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"