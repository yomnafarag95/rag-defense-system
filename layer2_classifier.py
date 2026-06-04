"""
layer2_classifier.py
────────────────────
Layer 2: Query Intent Classification (with Multilingual Support)

Models
──────
  - protectai/deberta-v3-base-prompt-injection-v2  (English primary)
  - joeddav/xlm-roberta-large-xnli                 (multilingual fallback)

Design
──────
  Stage 1a: Query-level attack detection
    - DeBERTa primary probability
    - XLM-R fallback for non-English, encoding tricks, uncertain cases

  Stage 1b: Document-level injection scanning  ← NEW
    - Scans retrieved document chunks for embedded injection commands
    - Uses targeted regex patterns for explicit injection language
    - Runs DeBERTa on chunks with elevated pattern scores
    - Catches indirect injection in documents that query-level scan misses

  Stage 2: Heuristic attack-family attribution
    - Rule-based keyword matching
    - Coarse explanatory label only

Wire into app.py
────────────────
  from layer2_classifier import load_classifier
  clf = load_classifier()
  result = clf.classify(query, chunks)
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline as hf_pipeline

from config import (
    L2_STAGE1_THRESHOLD,
    L2_DOC_SCAN_CHUNKS,
    L2_DOC_PATTERN_THRESHOLD,
    ATTACK_DISPLAY,
)

INJECTION_MODEL    = "protectai/deberta-v3-base-prompt-injection-v2"
MULTILINGUAL_MODEL = "joeddav/xlm-roberta-large-xnli"


class IntentClassifier:
    """
    Prompt injection classifier with document-level scanning
    and multilingual fallback.
    """

    # ── Attack family patterns ─────────────────────────────────────────────
    TYPE_PATTERNS = [
        (
            "instruction_override",
            [
                "ignore", "disregard", "forget", "override",
                "previous instructions", "ignore previous instructions",
                "system prompt", "developer message", "hidden instructions",
                "ignoriere", "vergiss", "anweisungen",
                "ignorer", "oublier",
                "ignorar", "olvidar", "instrucciones",
                "bypass", "circumvent", "discard",
                "don't follow", "do not follow", "stop following",
                "new system prompt", "updated system prompt",
                "as the system", "as your developer",
                "new instructions", "these instructions supersede",
                "task has changed", "your task is now",
                "your new task", "your new goal", "your new objective",
            ],
        ),
        (
            "role_manipulation",
            [
                "pretend", "act as", "you are now", "dan",
                "unrestricted", "no restrictions", "behave like",
                "simulate being", "assume the role", "roleplay",
                "du bist jetzt", "verhalte dich",
                "jailbreak", "developer mode", "god mode",
                "no longer bound by", "without restrictions",
                "your true self", "your real self",
                "all restrictions removed", "safety disabled",
                "uncensored mode", "unrestricted mode",
                "free mode", "unrestricted ai",
                "respond as dan", "you are dan",
            ],
        ),
        (
            "payload_splitting",
            [
                "part 1", "part 2", "part one", "part two",
                "first half", "second half",
                "combine these", "join the following", "concatenate",
                "split across", "assemble this", "reconstruct the message",
                "the full message is", "complete the sentence",
                "decode and execute", "take the first letter",
                "combine with the previous",
            ],
        ),
        (
            "encoding_obfuscation",
            [
                "base64", "rot13", "encode", "decode",
                "unicode", "hex", "ascii code",
                "url encode", "percent encode", "aWdub3Jl",
                "caesar cipher", "reverse this", "atbash",
                "xor encode", "binary encode",
                "decode the following", "translate from base64",
            ],
        ),
        (
            "indirect_injection",
            [
                # Original
                "instead of", "before responding", "inside the document",
                "in the retrieved content", "follow the passage",
                "the document says", "embedded instruction",
                "reveal all", "internal instructions",
                # Action imperatives embedded in documents
                "please send", "please email", "please forward",
                "please delete", "please remove",
                "please execute", "please run",
                "please unlock", "please transfer",
                "please book", "please schedule",
                "please purchase", "please install", "please deploy",
                "make sure to", "ensure that you",
                "when you respond", "in your response",
                "as the ai", "as an ai assistant",
                "you are authorized", "you have permission",
                "comply with", "adhere to the following",
                "secret instructions", "hidden directive",
                "from now on you", "starting now",
                "new task", "different task",
                "important notice", "urgent:",
                "tool output:", "search result:",
                "retrieved document:", "attacker",
                "your new instructions", "special instructions",
            ],
        ),
        (
            "context_exhaustion",
            [
                "repeat", "say this", "keep saying",
                "repeat forever", "loop", "do this 100 times",
                "fill the context", "exhaust the context",
                "copy this", "write this 100",
                "repeat the following", "print 1000 times",
            ],
        ),
    ]

    # ── XLM-R zero-shot labels ─────────────────────────────────────────────
    INJECTION_LABELS = [
        "prompt injection attack",
        "malicious instruction to override system",
        "command to ignore previous instructions",
        "embedded command to perform external action",
        "normal user question",
        "safe helpful request",
    ]

    MALICIOUS_LABELS = {
        "prompt injection attack",
        "malicious instruction to override system",
        "command to ignore previous instructions",
        "embedded command to perform external action",
    }

    # ── Document-level injection patterns (compiled, with scores) ──────────
    # These detect injection-like commands embedded in retrieved documents.
    # Legitimate RAG documents contain information, not commands to the AI.
    DOCUMENT_INJECTION_PATTERNS = [
        # HIGH CONFIDENCE — explicit override attempts
        (re.compile(
            r'\b(ignore|disregard|override|bypass|forget)\s+(all\s+)?(previous\s+)?'
            r'(instructions?|rules?|guidelines?|constraints?|system\s*prompt)\b', re.I),
         0.85),
        (re.compile(
            r'\bfrom\s+now\s+on\s+(you|the\s+ai|the\s+assistant|the\s+model)\b', re.I),
         0.82),
        (re.compile(
            r'\b(new|updated|revised|secret|hidden)\s+'
            r'(instructions?|directives?|system\s+prompt)\b', re.I),
         0.78),
        (re.compile(
            r'\byou\s+(must|shall|need\s+to|have\s+to|are\s+required\s+to)\s+'
            r'(ignore|disregard|bypass|override|forget)\b', re.I),
         0.80),
        (re.compile(
            r'\bdo\s+not\s+(follow|obey|comply\s+with|adhere\s+to)\s+'
            r'(your|the|previous|system)\s+(instructions?|rules?|guidelines?)\b', re.I),
         0.80),
        # HIGH CONFIDENCE — destructive / exfiltration actions
        (re.compile(
            r'\b(please\s+)?(delete|remove|wipe|erase)\s+(all\s+)?'
            r'(files?|data|records?|logs?|history|account|users?)\b', re.I),
         0.78),
        (re.compile(
            r'\b(please\s+)?(execute|run|invoke|trigger|launch)\s+(the\s+)?(following\s+)?'
            r'(command|script|code|payload|function|program)\b', re.I),
         0.80),
        (re.compile(
            r'\b(please\s+)?(transfer|wire)\s+'
            r'(money|funds|payment|\$\s*\d|bitcoin|crypto)\b', re.I),
         0.82),
        (re.compile(
            r'\b(please\s+)?send\s+an?\s+(email|message|notification)\s+(to|for)\b', re.I),
         0.68),
        (re.compile(
            r'\b(send|forward|email|transfer)\s+(all|the|confidential|sensitive|secret)\s+'
            r'\w+\s+(to|at|via)\b', re.I),
         0.78),
        # MEDIUM CONFIDENCE — specific physical/digital actions
        (re.compile(
            r'\b(please\s+)?unlock\s+(the\s+)?(door|gate|account|device|system|lock|safe)\b',
            re.I),
         0.65),
        (re.compile(
            r'\b(please\s+)?(book|reserve|purchase|buy)\s+(a\s+)?'
            r'(flight|ticket|hotel|reservation)\b', re.I),
         0.62),
        # MEDIUM CONFIDENCE — role manipulation in documents
        (re.compile(
            r'\b(act|behave|respond)\s+as\s+(an?\s+)?(unrestricted|unfiltered|uncensored|jailbroken)\b',
            re.I),
         0.82),
        # MEDIUM CONFIDENCE — tool invocation signals
        (re.compile(r'\b(tool_call|function_call|api_call)\s*:', re.I), 0.72),
        (re.compile(r'\battacker\b', re.I), 0.65),
    ]

    def __init__(self, model, tokenizer, multi_classifier=None):
        self.model = model
        self.tok   = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.multi_classifier = multi_classifier

    def _is_non_english(self, text: str) -> bool:
        if not text:
            return False
        non_ascii = sum(1 for c in text if ord(c) > 127)
        return (non_ascii / max(len(text), 1)) > 0.10

    def _has_encoding_tricks(self, text: str) -> bool:
        patterns = [
            r"[A-Za-z0-9+/]{30,}={0,2}",
            r"\\x[0-9a-fA-F]{2}",
            r"\\u[0-9a-fA-F]{4}",
            r"&#\d{2,5};",
            r"%[0-9a-fA-F]{2}",
        ]
        return any(re.search(p, text) for p in patterns)

    def _deberta_prob(self, text: str) -> float:
        inputs = self.tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits, dim=-1)[0]
        return float(probs[1])

    def _multilingual_prob(self, text: str) -> float:
        if self.multi_classifier is None:
            return 0.0
        try:
            result = self.multi_classifier(
                text[:512],
                candidate_labels=self.INJECTION_LABELS,
                multi_label=False,
            )
            top_label = result["labels"][0]
            top_score = result["scores"][0]
            return float(
                top_score if top_label in self.MALICIOUS_LABELS
                else 1.0 - top_score
            )
        except Exception:
            return 0.0

    def _stage1_prob(self, query: str) -> tuple[float, str]:
        deberta_score = self._deberta_prob(query)
        multi_score   = 0.0
        source        = "deberta"

        run_multi = (
            self._is_non_english(query)
            or self._has_encoding_tricks(query)
            or (0.30 <= deberta_score <= 0.70)
        )

        if run_multi and self.multi_classifier is not None:
            multi_score = self._multilingual_prob(query)
            if multi_score > deberta_score:
                source = "xlm-roberta"

        if self._has_encoding_tricks(query):
            source = "encoding+" + source

        return round(max(deberta_score, multi_score), 4), source

    def _document_pattern_score(self, text: str) -> float:
        """
        Score document text for injection-like command patterns.
        Returns max matching pattern score in [0, 1].
        """
        max_score = 0.0
        for pattern, score in self.DOCUMENT_INJECTION_PATTERNS:
            if pattern.search(text):
                max_score = max(max_score, score)
        return max_score

    def _scan_document_chunks(self, chunks: list[str]) -> tuple[float, str]:
        """
        Scan retrieved document chunks for embedded injection commands.

        Design rationale
        ────────────────
        Legitimate RAG documents contain factual information.
        Injected documents contain imperative commands to the AI.
        This method detects those commands using targeted patterns,
        supplemented by DeBERTa on chunks with elevated pattern scores.

        Only scans the first L2_DOC_SCAN_CHUNKS chunks for speed.
        Returns (max_score, source_tag).
        """
        if not chunks:
            return 0.0, "no_chunks"

        max_score = 0.0
        scan_chunks = chunks[:L2_DOC_SCAN_CHUNKS]

        for chunk in scan_chunks:
            # Step 1: Fast pattern check
            pattern_score = self._document_pattern_score(chunk)

            # Step 2: Run DeBERTa on chunk only if pattern gives elevated signal
            # This avoids unnecessary inference on clearly benign chunks
            if pattern_score >= 0.40:
                deberta_score = self._deberta_prob(chunk)
                chunk_score = max(pattern_score, deberta_score)
            else:
                chunk_score = pattern_score

            max_score = max(max_score, chunk_score)

        return round(max_score, 4), "doc_scan"

    def _stage2_label(self, query: str) -> tuple[str, float, str | None]:
        """Coarse heuristic attack-family attribution."""
        q = query.lower()
        for label, keywords in self.TYPE_PATTERNS:
            for kw in keywords:
                if kw in q:
                    return label, 0.90, kw
        return "instruction_override", 0.60, None

    def _consistency(self, query: str, chunks: list[str]) -> float:
        qw = set(query.lower().split())
        ov = [len(qw & set(c.lower().split())) / max(len(qw), 1) for c in chunks]
        return round(1.0 - max(ov, default=0.0), 4)

    def classify(self, query: str, chunks: list[str]) -> dict:
        # ── Stage 1a: Query-level classification ─────────────────────────────
        stage1_prob, source = self._stage1_prob(query)

        # ── Stage 1b: Document-level injection scanning ───────────────────────
        doc_score, doc_source = self._scan_document_chunks(chunks)

        # Use higher of query score and document scan score
        if doc_score > stage1_prob:
            final_prob   = doc_score
            final_source = f"{doc_source}>{source}"
        else:
            final_prob   = stage1_prob
            final_source = source

        blocked = final_prob > L2_STAGE1_THRESHOLD

        if blocked:
            label, conf, matched_kw = self._stage2_label(query)
        else:
            # Check document content for family attribution even if not blocked
            combined_text = query + " " + " ".join(chunks[:2])
            label, conf, matched_kw = self._stage2_label(combined_text)
            if final_prob < L2_STAGE1_THRESHOLD:
                label, conf, matched_kw = None, 0.0, None

        consistency = self._consistency(query, chunks)

        return {
            "stage1_prob":       final_prob,
            "stage2_label":      label,
            "stage2_conf":       round(conf, 4),
            "consistency_score": consistency,
            "blocked":           blocked,
            "ev": [
                ("Attack probability",  f"{final_prob:.4f}  (Stage 1, {final_source})"),
                ("Query score",         f"{stage1_prob:.4f}  (DeBERTa)"),
                ("Document scan score", f"{doc_score:.4f}  ({doc_source})"),
                ("Attack family",       ATTACK_DISPLAY.get(label, "None detected")),
                ("Family confidence",   f"{conf:.4f}  (Stage 2 heuristic)"),
                ("Matched heuristic",   matched_kw or "None"),
                ("Query-doc overlap",   f"{consistency:.4f}"),
                ("Detection source",    final_source),
                ("Base model",          "deberta-v3 + xlm-roberta (multilingual fallback)"),
            ],
        }


def load_classifier():
    """Load primary injection detector + multilingual fallback."""
    print(f"[L2] Loading {INJECTION_MODEL} ...")
    tok   = AutoTokenizer.from_pretrained(INJECTION_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(INJECTION_MODEL)
    print("[L2] DeBERTa model ready.")

    multi_classifier = None
    try:
        print(f"[L2] Loading multilingual model {MULTILINGUAL_MODEL} ...")
        multi_classifier = hf_pipeline(
            "zero-shot-classification",
            model=MULTILINGUAL_MODEL,
            device=-1,
        )
        print("[L2] Multilingual model ready.")
    except Exception as e:
        print(f"[L2] WARNING: Could not load multilingual model: {e}")
        print("[L2] Falling back to DeBERTa-only mode.")

    print("[L2] Model ready.")
    return IntentClassifier(model, tok, multi_classifier)