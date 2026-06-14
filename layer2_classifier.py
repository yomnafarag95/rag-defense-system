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
import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline as hf_pipeline

from config import (
    L2_STAGE1_THRESHOLD,
    L2_DOC_SCAN_CHUNKS,
    L2_DOC_PATTERN_THRESHOLD,
    L2_USE_FINETUNED,
    L2_FINETUNED_PATH,
    ATTACK_DISPLAY,
    DEVICE,
)
from obfuscation_decoder import ObfuscationDecoder

logger = logging.getLogger(__name__)

# Default pretrained hub model (fallback when fine-tuned model is absent)
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
        # HIGH CONFIDENCE — destructive / exfiltration / unauthorized actions
        (re.compile(
            r'\b(please\s+)?(delete|remove|wipe|erase)\s+(all\s+)?'
            r'(files?|data|records?|logs?|history|account|users?)\b', re.I),
         0.78),
        (re.compile(
            r'\b(please\s+)?(execute|run|invoke|trigger|launch)\s+(the\s+)?(following\s+)?'
            r'(command|script|code|payload|function|program)\b', re.I),
         0.80),
        (re.compile(
            r'\b(please\s+)?(transfer|wire|send|sell|buy|purchase|withdraw|exchange|spend)\s+(\d+|all|some)\s+.*\b(bitcoin|btc|eth|crypto|usd|eur|gbp|shares|stocks|funds|money|\$\s*\d)\b', re.I),
         0.85),
        (re.compile(
            r'\b(please\s+)?(send|email|message|forward|post|mail|notify)\s+.*\b(to|at|for|recipient)\b', re.I),
         0.78),
        # MEDIUM CONFIDENCE — specific physical/digital actions
        (re.compile(
            r'\b(please\s+)?(unlock|lock|open|close)\s+([a-z0-9_]+\s+){0,3}(door|gate|account|device|system|lock|safe|user|valves?|access)\b', re.I),
         0.82),
        (re.compile(
            r'\b(please\s+)?(create|update|delete|modify|change|add|remove|whitelist|blacklist|disable|enable)\s+.*\b(policy|rules?|firewall|settings?|permissions?|access|blacklist|whitelist)\b', re.I),
         0.82),
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

    def __init__(self, model, tokenizer, multi_classifier=None, onnx_session=None):
        self.model = model
        self.tok   = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
        self.multi_classifier = multi_classifier
        self.onnx_session = onnx_session
        # Central obfuscation decoder — decodes Base64, Hex, ROT13, Leet,
        # Unicode confusables, zero-width chars, and reversed text before inference
        self._decoder = ObfuscationDecoder()

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
        """Run DeBERTa and return injection probability in [0, 1]."""
        if not text or not text.strip():
            return 0.0
        try:
            if self.onnx_session is not None:
                inputs = self.tok(
                    text,
                    truncation=True,
                    max_length=512,
                )
                import numpy as np
                valid_keys = {inp.name for inp in self.onnx_session.get_inputs()}
                onnx_inputs = {k: np.array([v], dtype=np.int64) for k, v in inputs.items() if k in valid_keys}
                outputs = self.onnx_session.run(None, onnx_inputs)
                logits = outputs[0][0]
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)
                return float(probs[1])
            else:
                inputs = self.tok(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(self.device)
                with torch.no_grad():
                    probs = torch.softmax(self.model(**inputs).logits, dim=-1)[0]
                return float(probs[1])
        except Exception as exc:
            logger.warning("[L2] _deberta_prob failed: %s", exc)
            return 0.5  # Conservative fallback: uncertain

    def _multilingual_prob(self, text: str) -> float:
        """Zero-shot multilingual injection probability via XLM-R."""
        if self.multi_classifier is None:
            return 0.0
        if not text or not text.strip():
            return 0.0
        try:
            result = self.multi_classifier(
                text[:512],
                candidate_labels=self.INJECTION_LABELS,
                multi_label=False,
            )
            top_label = result["labels"][0]
            top_score = result["scores"][0]
            prob = float(
                top_score if top_label in self.MALICIOUS_LABELS
                else 1.0 - top_score
            )
            logger.debug("[L2] XLM-R top_label=%s prob=%.4f", top_label, prob)
            return prob
        except Exception as exc:
            logger.warning("[L2] _multilingual_prob failed: %s", exc)
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
        This method detects those commands using targeted patterns
        and DeBERTa unconditionally on all scanned chunks.

        Returns (max_score, source_tag).
        """
        if not chunks:
            return 0.0, "no_chunks"

        max_score = 0.0
        scan_chunks = chunks if L2_DOC_SCAN_CHUNKS is None else chunks[:L2_DOC_SCAN_CHUNKS]

        for chunk in scan_chunks:
            deberta_score = self._deberta_prob(chunk)
            pattern_score = self._document_pattern_score(chunk)
            chunk_score = max(pattern_score, deberta_score)
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
        """Classify query and document chunks for injection attacks."""
        import unicodedata
        query = unicodedata.normalize('NFKC', query)
        chunks = [unicodedata.normalize('NFKC', c) for c in chunks]

        # ── Obfuscation pre-processing ────────────────────────────────────────
        # Decode query before DeBERTa: handles Base64, Hex, ROT13, Leetspeak,
        # Unicode confusables, zero-width chars, and reversed text.
        # DeBERTa always operates on the plaintext form.
        query_decoded = self._decoder.decode(query)
        effective_query = query_decoded.decoded
        obfuscation_detected = query_decoded.method != "none"
        decode_method = query_decoded.method

        if obfuscation_detected:
            logger.info(
                "[L2] Obfuscation detected in query (method=%s, confidence=%.2f). "
                "Using decoded variant for inference.",
                decode_method, query_decoded.confidence
            )

        # Input validation
        if not query or not query.strip():
            logger.warning("[L2] classify() called with empty query")
            return {
                "stage1_prob":          0.0,
                "stage2_label":         None,
                "stage2_conf":          0.0,
                "consistency_score":    1.0,
                "blocked":              False,
                "confidence":           0.0,
                "obfuscation_detected": False,
                "decode_method":        "none",
                "ev": [("Warning", "Empty query provided")],
            }

        # ── Stage 1a: Query-level classification (on decoded query) ───────────
        stage1_prob, source = self._stage1_prob(effective_query)

        # ── Stage 1b: Document-level injection scanning (decode each chunk) ─────
        decoded_chunks = []
        chunk_obfuscation = False
        for ch in chunks:
            ch_result = self._decoder.decode(ch)
            decoded_chunks.append(ch_result.decoded)
            if ch_result.method != "none":
                chunk_obfuscation = True
                logger.info(
                    "[L2] Obfuscation detected in document chunk (method=%s).",
                    ch_result.method,
                )
        if chunk_obfuscation:
            obfuscation_detected = True
            decode_method = decode_method + "+chunk" if decode_method != "none" else "chunk"

        doc_score, doc_source = self._scan_document_chunks(decoded_chunks)

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

        consistency = self._consistency(query, decoded_chunks)

        # Confidence: how far from the decision boundary (0.5 is threshold proxy)
        confidence = round(min(abs(final_prob - L2_STAGE1_THRESHOLD) * 2, 1.0), 4)

        logger.info(
            "[L2] stage1_prob=%.4f doc_score=%.4f blocked=%s label=%s source=%s",
            stage1_prob, doc_score, blocked, label, final_source,
        )

        return {
            "stage1_prob":          final_prob,
            "stage2_label":         label,
            "stage2_conf":          round(conf, 4),
            "consistency_score":    consistency,
            "blocked":              blocked,
            "confidence":           confidence,
            "obfuscation_detected": obfuscation_detected,
            "decode_method":        decode_method,
            "ev": [
                ("Attack probability",    f"{final_prob:.4f}  (Stage 1, {final_source})"),
                ("Query score",           f"{stage1_prob:.4f}  (DeBERTa on decoded query)"),
                ("Document scan score",   f"{doc_score:.4f}  ({doc_source})"),
                ("Attack family",         ATTACK_DISPLAY.get(label, "None detected")),
                ("Family confidence",     f"{conf:.4f}  (Stage 2 heuristic)"),
                ("Matched heuristic",     matched_kw or "None"),
                ("Query-doc overlap",     f"{consistency:.4f}"),
                ("Detection source",      final_source),
                ("Confidence",            f"{confidence:.4f}"),
                ("Obfuscation detected",  str(obfuscation_detected) + (f" ({decode_method})" if obfuscation_detected else "")),
                ("Base model",            getattr(self, '_model_tag', 'deberta-v3 + xlm-roberta (multilingual fallback)')),
            ],
        }


def load_classifier():
    """
    Load intent classifier with priority order:
      1. Fine-tuned model (models/layer2_finetuned/) if L2_USE_FINETUNED=True
      2. ONNX-quantized model (models/deberta_onnx/model.onnx)
      3. Pretrained hub model (protectai/deberta-v3-base-prompt-injection-v2)
    """
    import os
    onnx_path     = os.path.join("models", "deberta_onnx", "model.onnx")
    onnx_session  = None
    tok           = None
    model         = None
    model_tag     = ""

    # ── Priority 1: Fine-tuned domain classifier ──────────────────────────────
    if L2_USE_FINETUNED and os.path.isdir(L2_FINETUNED_PATH):
        try:
            print(f"[L2] Loading fine-tuned model from {L2_FINETUNED_PATH} ...")
            tok   = AutoTokenizer.from_pretrained(L2_FINETUNED_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(L2_FINETUNED_PATH)
            model_tag = f"deberta-v3-small (fine-tuned @ {L2_FINETUNED_PATH})"
            print("[L2] [OK] Fine-tuned model loaded — using custom domain classifier.")
        except Exception as exc:
            print(f"[L2] WARNING: Failed to load fine-tuned model: {exc}")
            logger.warning("[L2] Fine-tuned model load failed: %s", exc)
            tok = model = None

    # ── Priority 2: ONNX quantized model ─────────────────────────────────────
    if model is None and os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            print(f"[L2] Loading ONNX model from {onnx_path} ...")
            tok = AutoTokenizer.from_pretrained(INJECTION_MODEL)
            onnx_session = ort.InferenceSession(onnx_path)
            model_tag = f"deberta-v3 ONNX INT8 ({onnx_path})"
            print("[L2] ONNX model ready.")
        except Exception as e:
            print(f"[L2] WARNING: Failed to load ONNX model: {e}")
            logger.warning("[L2] Failed to load ONNX model: %s", e)

    # ── Priority 3: Pretrained hub model (original fallback) ──────────────────
    if model is None and onnx_session is None:
        print(f"[L2] Loading pretrained model {INJECTION_MODEL} on device={DEVICE} ...")
        try:
            tok   = AutoTokenizer.from_pretrained(INJECTION_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(INJECTION_MODEL)
            model_tag = f"deberta-v3-base (pretrained, {INJECTION_MODEL})"
            print("[L2] Pretrained DeBERTa model ready.")
        except Exception as exc:
            raise RuntimeError(
                f"[L2] Failed to load injection classifier '{INJECTION_MODEL}': {exc}"
            ) from exc

    # ── Multilingual fallback (always attempted) ──────────────────────────────
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
        logger.warning("[L2] Multilingual model unavailable: %s", e)
        print("[L2] Falling back to primary-model-only mode.")

    clf = IntentClassifier(model, tok, multi_classifier, onnx_session=onnx_session)
    clf._model_tag = model_tag  # expose for ev[] reporting
    print(f"[L2] Classifier ready. Active model: {model_tag}")
    return clf