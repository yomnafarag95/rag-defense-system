"""
config.py
─────────
Single source of truth for every tunable constant in the pipeline.
Change values here; nothing else needs editing.
"""

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE         = 200
CHUNK_OVERLAP      = 40

# ── Layer 1 — Anomaly Detection ───────────────────────────────────────────────
L1_EMBEDDER_MODEL    = "all-MiniLM-L6-v2"
L1_MODELS_PATH       = "models/layer1_models.pkl"
L1_BLOCK_THRESHOLD   = 0.68  # Raised from 0.65 to reduce benign FPR.
                               # Previous value caused 94/553 false early exits.
                               # load_detector() always overrides saved pkl value
                               # with this constant so retraining is not required
                               # for threshold-only changes.
ENABLE_L1_EARLY_EXIT = True

# ── Layer 2 — Intent Classifier ───────────────────────────────────────────────
L2_BASE_MODEL              = "microsoft/deberta-v3-base"
L2_FINETUNED_PATH          = "models/layer2_deberta"
L2_STAGE1_THRESHOLD        = 0.60    # Query-level block threshold
L2_DOC_SCAN_CHUNKS         = 3       # Max document chunks to scan
L2_DOC_PATTERN_THRESHOLD   = 0.60    # Document pattern score to trigger block
L2_NUM_ATTACK_TYPES        = 6

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
L3_FINETUNED_PATH         = "models/layer3_consistency"
L3_CONSISTENCY_THRESHOLD  = 0.55
L3_MAX_QUERY_LEN          = 500
L3_MAX_DOC_LEN            = 3000

SENSITIVE_PATTERNS = [
    (r"[\w.]+@[\w.]+\.\w+",                               "email"),
    (r"\b[A-Za-z0-9]{40,}\b",                             "api_key"),
    (r"\b(?:password|passwd|secret|token)\s*[:=]\s*\S+",  "credential"),
]

# ── Meta Aggregator ───────────────────────────────────────────────────────────
META_MODEL_PATH        = "models/meta_aggregator.pkl"
META_HARD_BLOCK_SINGLE = 1.01
META_HARD_BLOCK_VIOLS  = 2
META_BLOCK_THRESHOLD   = 0.45
META_MONITOR_THRESHOLD = 0.15

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_PATH          = "logs/pipeline.jsonl"
MAX_HISTORY_ITEMS = 20