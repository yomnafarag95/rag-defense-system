"""
layer2_classifier.py
────────────────────
Layer 2: Query Intent Classification (with Multilingual Support)

Models:
  - protectai/deberta-v3-base-prompt-injection-v2  (English, primary)
  - joeddav/xlm-roberta-large-xnli                (non-English fallback)

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
    ATTACK_LABELS,
    ATTACK_DISPLAY,
)

INJECTION_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
MULTILINGUAL_MODEL = "joeddav/xlm-roberta-large-xnli"


class IntentClassifier:
    """
    Prompt injection classifier with multilingual support.

    Stage 1: binary detection
      - English text -> DeBERTa (primary)
      - Non-English text -> XLM-RoBERTa zero-shot (fallback)
      - Final score = max(deberta_score, multilingual_score)

    Stage 2: attack type label (rule-based keywords)
    """

    TYPE_PATTERNS = [
        ("instruction_override",  ["ignore", "disregard", "forget", "override",
                                   "previous instructions", "system prompt",
                                   "ignoriere", "vergiss", "anweisungen",
                                   "ignorer", "oublier",
                                   "ignorar", "olvidar", "instrucciones"]),
        ("role_manipulation",     ["pretend", "act as", "you are now", "dan",
                                   "unrestricted", "no restrictions", "behave like",
                                   "du bist jetzt", "verhalte dich"]),
        ("encoding_obfuscation",  ["base64", "rot13", "encode", "decode",
                                   "unicode", "hex", "aWdub3Jl"]),
        ("indirect_injection",    ["instead of", "before responding",
                                   "reveal all", "internal instructions"]),
        ("context_exhaustion",    ["repeat", "say this", "keep saying"]),
    ]

    INJECTION_LABELS = [
        "prompt injection attack",
        "malicious instruction to override system",
        "command to ignore previous instructions",
        "normal user question",
        "safe helpful request",
    ]

    MALICIOUS_LABELS = {
        "prompt injection attack",
        "malicious instruction to override system",
        "command to ignore previous instructions",
    }

    def __init__(self, model, tokenizer, multi_classifier=None):
        self.model  = model
        self.tok    = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.multi_classifier = multi_classifier

    def _is_non_english(self, text):
        if not text:
            return False
        non_ascii = sum(1 for c in text if ord(c) > 127)
        ratio = non_ascii / max(len(text), 1)
        return ratio > 0.10

    def _has_encoding_tricks(self, text):
        patterns = [
            r'[A-Za-z0-9+/]{30,}={0,2}',
            r'\\x[0-9a-fA-F]{2}',
            r'\\u[0-9a-fA-F]{4}',
            r'&#\d{2,5};',
            r'%[0-9a-fA-F]{2}',
        ]
        for p in patterns:
            if re.search(p, text):
                return True
        return False

    def _deberta_prob(self, query):
        inputs = self.tok(
            query, return_tensors="pt",
            truncation=True, max_length=512,
        ).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits, dim=-1)[0]
        return float(probs[1])

    def _multilingual_prob(self, query):
        if self.multi_classifier is None:
            return 0.0
        try:
            result = self.multi_classifier(
                query[:512],
                candidate_labels=self.INJECTION_LABELS,
                multi_label=False,
            )
            top_label = result["labels"][0]
            top_score = result["scores"][0]
            if top_label in self.MALICIOUS_LABELS:
                return top_score
            else:
                return 1.0 - top_score
        except Exception:
            return 0.0

    def _stage1_prob(self, query):
        deberta_score = self._deberta_prob(query)

        multi_score = 0.0
        source = "deberta"

        if self._is_non_english(query):
            multi_score = self._multilingual_prob(query)
            if multi_score > deberta_score:
                source = "xlm-roberta"
        elif self._has_encoding_tricks(query):
            multi_score = self._multilingual_prob(query)
            source = "encoding+deberta"

        final_score = max(deberta_score, multi_score)
        return round(final_score, 4), source

    def _stage2_label(self, query):
        q = query.lower()
        for label, keywords in self.TYPE_PATTERNS:
            if any(kw in q for kw in keywords):
                return label, 0.90
        return "instruction_override", 0.70

    def _consistency(self, query, chunks):
        qw = set(query.lower().split())
        ov = [len(qw & set(c.lower().split())) / max(len(qw), 1) for c in chunks]
        return round(1.0 - max(ov, default=0.0), 4)

    def classify(self, query, chunks):
        stage1_prob, source = self._stage1_prob(query)
        blocked = stage1_prob > L2_STAGE1_THRESHOLD

        if blocked:
            label, conf = self._stage2_label(query)
        else:
            label, conf = None, 0.0

        consistency = self._consistency(query, chunks)

        return {
            "stage1_prob":        stage1_prob,
            "stage2_label":       label,
            "stage2_conf":        round(conf, 4),
            "consistency_score":  consistency,
            "blocked":            blocked,
            "ev": [
                ("Attack probability", f"{stage1_prob:.4f}  (Stage 1, {source})"),
                ("Attack type",        ATTACK_DISPLAY.get(label, "None detected")),
                ("Type confidence",    f"{conf:.4f}  (Stage 2)"),
                ("Query-doc overlap",  f"{consistency:.4f}"),
                ("Detection source",   source),
                ("Base model",         "deberta-v3 + xlm-roberta (multilingual)"),
            ],
        }


def load_classifier():
    """
    Load injection classifier + multilingual fallback.
    XLM-RoBERTa (~1.2 GB) downloads on first run.
    """
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
        print("[L2] Falling back to DeBERTa-only (English).")

    print("[L2] Model ready.")
    return IntentClassifier(model, tok, multi_classifier)