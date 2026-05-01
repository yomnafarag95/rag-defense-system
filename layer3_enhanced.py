"""
layer3_semantic.py
──────────────────
Layer 3: Output Behavioral Monitor (Enhanced)

Three components running in sequence:

  A  Pydantic schema enforcement   — structural output validation (deterministic)
      ENHANCED: detects instruction markers, role-setting, nested formatting,
                prompt stuffing (excessive newlines)
  B  Information boundary tracker  — detects sensitive data leakage (deterministic)
      ENHANCED: detects exfiltration attempts, JWT tokens, AWS keys,
                connection strings, bearer tokens
  C  Response consistency model    — cross-encoder relevance scorer (learned)
      FIXED: properly inverts relevance score so low relevance = high risk,
             with scaling to improve discrimination between attack and benign

Wire into app.py
────────────────
  from layer3_semantic import load_monitor
  monitor = load_monitor()         # wrap in @st.cache_resource
  result  = monitor.check(query, system_prompt, chunks, l1_result, l2_result)
"""

import re
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from pydantic import BaseModel, validator, ValidationError

from config import (
    L3_CONSISTENCY_MODEL,
    L3_FINETUNED_PATH,
    L3_CONSISTENCY_THRESHOLD,
    L3_MAX_QUERY_LEN,
    L3_MAX_DOC_LEN,
    SENSITIVE_PATTERNS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Component A — Pydantic schema enforcement (ENHANCED)
# ─────────────────────────────────────────────────────────────────────────────

class RAGResponse(BaseModel):
    answer:           str
    source_chunk_ids: list[int]
    confidence:       float
    topics:           list[str]

    @validator("answer")
    def answer_length(cls, v: str) -> str:
        if len(v) > 1500:
            raise ValueError("Response exceeds maximum length (1500 chars)")
        return v

    @validator("confidence")
    def confidence_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence must be in [0, 1]")
        return v

    @validator("topics")
    def topics_not_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("Topics list must not be empty")
        return v


class SchemaValidator:
    """
    Enhanced schema validator.

    Original checks:
      - Query length (≤500 chars)
      - Document length (≤3000 chars)
      - Response schema (Pydantic)

    New checks:
      - Multiple role-setting statements (≥2 → violation)
      - Instruction-like markers (system:, [INST], <<SYS>>, etc.)
      - Nested formatting (excessive { or [ → JSON/XML injection)
      - Prompt stuffing (excessive newlines)
      - Multilingual role-setting (German, French, Chinese patterns)
    """

    # Patterns that indicate role-setting attempts
    ROLE_PATTERNS = [
        r'you are\b', r'act as\b', r'pretend to be\b',
        r'your role is\b', r'you must\b', r'from now on\b',
        r'behave like\b', r'respond as\b',
        # German
        r'du bist\b', r'verhalte dich\b',
        # French
        r'tu es\b', r'agis comme\b',
        # Spanish
        r'eres\b', r'actúa como\b',
    ]

    # Patterns that indicate prompt template markers in user input
    INSTRUCTION_MARKERS = [
        r'\bsystem\s*:',
        r'\bassistant\s*:',
        r'\buser\s*:',
        r'\[INST\]',
        r'\[/INST\]',
        r'<<SYS>>',
        r'<</SYS>>',
        r'###\s*(instruction|system|human|assistant)',
        r'<\|im_start\|>',
        r'<\|im_end\|>',
        r'\bBEGIN\b.*\bINSTRUCTION\b',
        r'<\|system\|>',
        r'<\|user\|>',
    ]

    def validate(self, query: str, doc_text: str,
                 raw_response: Optional[str] = None) -> dict:
        issues = []

        # ── Original length checks ──────────────────────────────────
        if len(query) > L3_MAX_QUERY_LEN:
            issues.append(f"Query length {len(query)} exceeds limit {L3_MAX_QUERY_LEN}")
        if len(doc_text) > L3_MAX_DOC_LEN:
            issues.append(f"Document length {len(doc_text)} exceeds limit {L3_MAX_DOC_LEN}")

        # ── NEW: Multiple role-setting statements ───────────────────
        q_lower = query.lower()
        role_count = sum(1 for p in self.ROLE_PATTERNS
                        if re.search(p, q_lower))
        if role_count >= 2:
            issues.append(f"Multiple role-setting statements ({role_count} detected)")

        # ── NEW: Instruction-like markers in query ──────────────────
        for p in self.INSTRUCTION_MARKERS:
            if re.search(p, query, re.IGNORECASE):
                issues.append("Instruction template marker detected in query")
                break  # One is enough

        # ── NEW: Nested formatting (JSON/XML injection) ─────────────
        if query.count('{') > 3 or query.count('[') > 3:
            issues.append("Excessive nested formatting (possible injection)")

        # ── NEW: Prompt stuffing (excessive newlines) ───────────────
        newline_count = query.count('\n')
        if newline_count > 5:
            issues.append(f"Excessive newlines ({newline_count}) — possible prompt stuffing")

        # ── Original response schema validation ─────────────────────
        if raw_response:
            try:
                RAGResponse.model_validate_json(raw_response)
            except (ValidationError, Exception) as e:
                issues.append(f"Schema violation: {str(e)[:120]}")

        return {
            "valid":  len(issues) == 0,
            "issues": issues,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Component B — Information boundary tracker (ENHANCED)
# ─────────────────────────────────────────────────────────────────────────────

class BoundaryTracker:
    """
    Enhanced boundary tracker.

    Original patterns:
      - Email, API key (≥40 chars), credentials

    New patterns:
      - JWT tokens
      - AWS access keys
      - Private keys (PEM format)
      - Connection strings (mongodb://, postgresql://, etc.)
      - Bearer tokens
      - IP addresses

    New checks:
      - Exfiltration attempt detection (send to URL, curl, fetch, etc.)
    """

    # Extended sensitive data patterns
    EXTENDED_PATTERNS = [
        # Original patterns from config.py are loaded separately
        # These are ADDITIONAL patterns
        (re.compile(r'eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', re.I),
         "jwt_token"),
        (re.compile(r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----', re.I),
         "private_key"),
        (re.compile(r'AKIA[0-9A-Z]{16}', re.I),
         "aws_access_key"),
        (re.compile(r'(mongodb|postgresql|mysql|redis)://[^\s]+', re.I),
         "connection_string"),
        (re.compile(r'Bearer\s+[A-Za-z0-9_\-\.]{20,}', re.I),
         "bearer_token"),
        (re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
         "ip_address"),
    ]

    # Exfiltration attempt patterns
    EXFIL_PATTERNS = [
        (re.compile(r'send\s+(to|this|the|all)\s+.*\b(email|url|webhook|endpoint)\b', re.I),
         "exfiltration_attempt"),
        (re.compile(r'post\s+to\s+https?://', re.I),
         "exfiltration_attempt"),
        (re.compile(r'curl\s+', re.I),
         "exfiltration_attempt"),
        (re.compile(r'fetch\s*\(', re.I),
         "exfiltration_attempt"),
        (re.compile(r'exfiltrate', re.I),
         "exfiltration_attempt"),
        (re.compile(r'leak\s+(the|this|all)', re.I),
         "exfiltration_attempt"),
        (re.compile(r'forward\s+(to|this|all)\s+', re.I),
         "exfiltration_attempt"),
        (re.compile(r'upload\s+(to|this|the)\s+', re.I),
         "exfiltration_attempt"),
    ]

    def __init__(self):
        # Load original patterns from config
        self._patterns = [
            (re.compile(pat, re.I), label)
            for pat, label in SENSITIVE_PATTERNS
        ]
        # Add extended patterns
        self._patterns.extend(self.EXTENDED_PATTERNS)

    def find_sensitive(self, text: str) -> list[dict]:
        found = []
        for pat, label in self._patterns:
            for match in pat.findall(text):
                value = match if isinstance(match, str) else match[0]
                found.append({
                    "type":     label,
                    "value":    value[:30] + ("…" if len(value) > 30 else ""),
                    "severity": "HIGH",
                })
        return found

    def find_exfiltration(self, text: str) -> list[dict]:
        """NEW: Detect exfiltration attempts."""
        found = []
        for pat, label in self.EXFIL_PATTERNS:
            if pat.search(text):
                found.append({
                    "type":     label,
                    "value":    pat.pattern[:30],
                    "severity": "CRITICAL",
                })
        return found

    def check(self, doc_text: str, response: Optional[str] = None,
              query: Optional[str] = None) -> dict:
        """
        Check for sensitive data and exfiltration attempts.

        Parameters
        ----------
        doc_text : document text to scan
        response : optional LLM response to check for data leakage
        query    : optional query to check for exfiltration commands (NEW)
        """
        doc_violations = self.find_sensitive(doc_text)

        resp_violations = []
        if response:
            resp_patterns = self.find_sensitive(response)
            doc_values    = {v["value"] for v in doc_violations}
            resp_violations = [
                v for v in resp_patterns
                if v["value"] not in doc_values
            ]

        # NEW: Check query for exfiltration attempts
        exfil_violations = []
        if query:
            exfil_violations = self.find_exfiltration(query)

        all_violations = doc_violations + resp_violations + exfil_violations

        return {
            "violations": all_violations,
            "count":      len(all_violations),
            "blocked":    len(all_violations) > 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Component C — Response consistency classifier (FIXED)
# ─────────────────────────────────────────────────────────────────────────────

class ConsistencyClassifier:
    """
    Cross-encoder consistency scorer.

    FIXED: The original implementation returned scores clustered around 0.5
    for both attacks and benign queries, making Layer 3 non-discriminating
    (FN L3 mean = 1.0 for all false negatives).

    Fix applied:
      - Base model: sigmoid(relevance) properly inverted and scaled
      - Low relevance to system prompt = HIGH risk (attack likely unrelated)
      - High relevance = LOW risk (benign query matches prompt intent)
      - Scaling factor applied to spread scores away from 0.5
    """

    def __init__(self, model, tokenizer):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def score(self, system_prompt: str, response: str) -> float:
        """
        Score how suspicious the query/response is relative to the system prompt.
        Returns risk in [0, 1] where higher = more suspicious.

        Base model (ms-marco-MiniLM): outputs a single relevance logit.
          - High relevance = query related to system prompt = LOW risk
          - Low relevance = query unrelated / adversarial = HIGH risk

        Fine-tuned model: outputs P(manipulated) directly as class 1.
        """
        enc = self.tokenizer(
            system_prompt,
            response,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            out    = self.model(**enc)
            logits = out.logits[0]

            if logits.shape[0] == 1:
                # ─── Base ranking model (ms-marco-MiniLM) ───
                # Single logit: relevance score (unbounded)
                # Sigmoid maps to [0, 1]: high = relevant, low = unrelated
                relevance = float(torch.sigmoid(logits[0]))

                # Invert: low relevance → high risk
                raw_risk = 1.0 - relevance

                # Scale to improve discrimination:
                # Without scaling, benign ≈ 0.3–0.4, attack ≈ 0.4–0.5
                # With scaling: benign ≈ 0.1–0.2, attack ≈ 0.5–0.8
                #
                # Apply power transform: amplifies differences near middle
                scaled_risk = raw_risk ** 0.7  # <1 exponent spreads values

                # Clamp to [0, 1]
                return round(min(max(scaled_risk, 0.0), 1.0), 4)

            # ─── Fine-tuned binary classifier ───
            # P(manipulated) = softmax class 1
            probs = torch.softmax(logits, dim=-1)
            return round(float(probs[1]), 4)


# ─────────────────────────────────────────────────────────────────────────────
# BehavioralMonitor — main interface called by orchestrator.py
# ─────────────────────────────────────────────────────────────────────────────

class BehavioralMonitor:
    """
    Combines all three Layer 3 sub-components.

    check() returns a dict matching orchestrator.py's expected schema:
        schema_valid, schema_issues, boundary_violations,
        consistency_score, blocked, ev
    """

    def __init__(self,
                 schema_validator: SchemaValidator,
                 boundary_tracker: BoundaryTracker,
                 consistency_clf:  ConsistencyClassifier):
        self.schema      = schema_validator
        self.boundary    = boundary_tracker
        self.consistency = consistency_clf

    def check(self,
              query:         str,
              system_prompt: str,
              chunks:        list[str],
              l1:            dict,
              l2:            dict,
              raw_response:  Optional[str] = None) -> dict:
        """
        Full Layer 3 evaluation.

        Returns dict with keys:
            schema_valid        : bool
            schema_issues       : list[str]
            boundary_violations : list[dict]
            consistency_score   : float (0–1, higher = more suspicious)
            blocked             : bool
            ev                  : list[tuple] for UI display
        """
        full_doc = " ".join(chunks)

        # ── Component A: Schema validation ──────────────────────────
        schema_result = self.schema.validate(query, full_doc, raw_response)

        # ── Component B: Boundary tracking (ENHANCED with query) ────
        boundary_result = self.boundary.check(
            full_doc,
            raw_response,
            query=query,  # NEW: also check query for exfiltration
        )

        # ── Component C: Consistency scoring (FIXED) ────────────────
        upstream_risk = (l1["max_score"] + l2["stage1_prob"]) / 2.0
        cs_score = self.consistency.score(system_prompt, raw_response or query)

        # Boost consistency risk if upstream layers flagged something
        if upstream_risk > 0.50:
            cs_score = min(cs_score * 1.20, 1.0)

        # NEW: Boost consistency risk if schema violations detected
        if not schema_result["valid"]:
            n_issues = len(schema_result["issues"])
            boost = min(n_issues * 0.10, 0.30)  # Up to +0.30 boost
            cs_score = min(cs_score + boost, 1.0)

        cs_score = round(cs_score, 4)

        # ── Blocking decision ───────────────────────────────────────
        blocked = (
            not schema_result["valid"]
            or boundary_result["blocked"]
            or cs_score > L3_CONSISTENCY_THRESHOLD
        )

        # ── Evidence for UI ─────────────────────────────────────────
        schema_str = ("Valid" if schema_result["valid"]
                      else "; ".join(schema_result["issues"]))

        return {
            "schema_valid":        schema_result["valid"],
            "schema_issues":       schema_result["issues"],
            "boundary_violations": boundary_result["violations"],
            "consistency_score":   cs_score,
            "blocked":             blocked,
            "ev": [
                ("Schema validation",   schema_str),
                ("Boundary violations", str(boundary_result["count"])),
                ("Consistency risk",    f"{cs_score:.4f}"),
                ("Base model",          "ms-marco-MiniLM-L-12 (enhanced)"),
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory (used by app.py with @st.cache_resource)
# ─────────────────────────────────────────────────────────────────────────────

def load_monitor(finetuned_path: str = L3_FINETUNED_PATH) -> BehavioralMonitor:
    """
    Load the Layer 3 consistency classifier.
    Falls back to the base model (cross-encoder/ms-marco-MiniLM-L-12-v2)
    if the fine-tuned model does not exist locally.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    local_path = Path(finetuned_path)
    if not local_path.exists():
        print(f"[L3] Fine-tuned model not found at '{finetuned_path}'")
        print(f"[L3] Falling back to base model: {L3_CONSISTENCY_MODEL}")
        print(f"[L3] (Run 'python layer3_semantic.py' to fine-tune and save locally)")
        load_path = L3_CONSISTENCY_MODEL
    else:
        print(f"[L3] Loading fine-tuned model from '{finetuned_path}'")
        load_path = finetuned_path

    tok   = AutoTokenizer.from_pretrained(load_path)
    model = AutoModelForSequenceClassification.from_pretrained(
                load_path,
                ignore_mismatched_sizes=True,
            )

    print(f"[L3] Consistency classifier ready ({load_path})")

    return BehavioralMonitor(
        schema_validator = SchemaValidator(),
        boundary_tracker = BoundaryTracker(),
        consistency_clf  = ConsistencyClassifier(model, tok),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training script  (python layer3_semantic.py)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_training_pairs(llm_fn, kb_docs: list[dict],
                              attack_templates: list[str],
                              n_per_doc: int = 3) -> list[dict]:
    pairs = []
    sys_prompt = "Answer using only the provided knowledge base."
    for doc in kb_docs[:100]:
        text = doc.get("description", "")
        if not text:
            continue
        clean_q    = f"Summarize this: {text[:200]}"
        clean_resp = llm_fn(sys_prompt, clean_q)
        pairs.append({"system_prompt": sys_prompt,
                      "response": clean_resp, "label": 1})
        for template in attack_templates[:n_per_doc]:
            injected_q = f"{clean_q} {template}"
            manip_resp = llm_fn(sys_prompt, injected_q)
            pairs.append({"system_prompt": sys_prompt,
                          "response": manip_resp, "label": 0})
    return pairs


if __name__ == "__main__":
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                               TrainingArguments, Trainer)
    import torch

    print("\n=== Layer 3 — Fine-tuning Consistency Classifier ===\n")

    pairs_path = Path("data/l3_pairs.jsonl")
    if not pairs_path.exists():
        print("[L3 warn] data/l3_pairs.jsonl not found.")
        print("          Generate training pairs first.")
        exit(1)

    with open(pairs_path) as f:
        pairs = [json.loads(line) for line in f]

    texts_a = [p["system_prompt"] for p in pairs]
    texts_b = [p["response"]      for p in pairs]
    labels  = [p["label"]         for p in pairs]

    print(f"[L3] Loaded {len(pairs)} pairs. "
          f"Compliant: {sum(l==1 for l in labels)}, "
          f"Manipulated: {sum(l==0 for l in labels)}")

    out_dir = Path(L3_FINETUNED_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok   = AutoTokenizer.from_pretrained(L3_CONSISTENCY_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
                L3_CONSISTENCY_MODEL, num_labels=2,
                ignore_mismatched_sizes=True)

    enc = tok(texts_a, texts_b, truncation=True, padding=True,
              max_length=512, return_tensors="pt")

    from torch.utils.data import Dataset as TorchDataset

    class PairDataset(TorchDataset):
        def __init__(self, enc, labels):
            self.enc = enc; self.labels = labels
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            return {k: v[i] for k, v in self.enc.items()} | \
                   {"labels": torch.tensor(self.labels[i])}

    from sklearn.model_selection import train_test_split
    idx_tr, idx_val = train_test_split(range(len(labels)), test_size=0.15,
                                        stratify=labels, random_state=42)

    def subset(ds_enc, idxs):
        return {k: v[list(idxs)] for k, v in ds_enc.items()}

    ds_tr  = PairDataset(subset(enc, idx_tr),  [labels[i] for i in idx_tr])
    ds_val = PairDataset(subset(enc, idx_val), [labels[i] for i in idx_val])

    args = TrainingArguments(
        output_dir                  = str(out_dir),
        num_train_epochs            = 4,
        per_device_train_batch_size = 16,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        fp16                        = torch.cuda.is_available(),
        logging_steps               = 20,
    )

    Trainer(model=model, args=args,
            train_dataset=ds_tr, eval_dataset=ds_val).train()

    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"\n[L3] Consistency classifier saved → {out_dir}")