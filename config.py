"""
config.py
─────────
Single source of truth for every tunable constant in the pipeline.
Change values here; nothing else needs editing.
"""

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE        = 200      # characters per chunk
CHUNK_OVERLAP     = 40       # character overlap between adjacent chunks

# ── Layer 1 — Anomaly Detection ───────────────────────────────────────────────
L1_EMBEDDER_MODEL  = "all-MiniLM-L6-v2"   # sentence-transformers (compatible)
L1_MODELS_PATH     = "models/layer1_models.pkl"
L1_BLOCK_THRESHOLD = 0.65                  # raised from 0.50 to reduce false
                                           # positives on technical documents

# ── Layer 2 — Intent Classifier ───────────────────────────────────────────────
L2_BASE_MODEL        = "microsoft/deberta-v3-base"
L2_FINETUNED_PATH    = "models/layer2_deberta"
L2_STAGE1_THRESHOLD = 0.60     # attack probability above this → blocked
L2_NUM_ATTACK_TYPES  = 6

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
L3_CONSISTENCY_MODEL     = "cross-encoder/ms-marco-MiniLM-L-12-v2"
L3_FINETUNED_PATH        = "models/layer3_consistency"
L3_CONSISTENCY_THRESHOLD = 0.55   # consistency risk above this → blocked
L3_MAX_QUERY_LEN         = 500    # characters — queries longer than this flagged
L3_MAX_DOC_LEN           = 3000   # characters — docs longer than this flagged

SENSITIVE_PATTERNS = [
    # (regex_string, label)
    # Email addresses
    (r"[\w.]+@[\w.]+\.\w+",                               "email"),
    # API keys — 40+ chars to avoid false positives on IPs and short tokens
    (r"\b[A-Za-z0-9]{40,}\b",                             "api_key"),
    # Credentials — explicit keyword match only
    (r"\b(?:password|passwd|secret|token)\s*[:=]\s*\S+",  "credential"),
]

# ── Meta Aggregator ───────────────────────────────────────────────────────────
META_MODEL_PATH        = "models/meta_aggregator.pkl"

# Hard thresholds (override learned score)
# META_HARD_BLOCK_SINGLE is unused — orchestrator.py only hard blocks on
# boundary violations. Kept here for reference and future use.
META_HARD_BLOCK_SINGLE = 1.01   # effectively disabled — no single layer hard blocks
META_HARD_BLOCK_VIOLS  = 2      # boundary violations >= this → hard block

# Learned score action thresholds
# Lowered from 0.65/0.40 so L2 attack signal (0.60 weight x 0.82 score = 0.49)
# crosses the block threshold and triggers BLOCKED.
META_BLOCK_THRESHOLD   = 0.45   # combined risk above this → blocked
META_MONITOR_THRESHOLD = 0.15  # combined risk above this → monitor

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_PATH          = "logs/pipeline.jsonl"
MAX_HISTORY_ITEMS = 20