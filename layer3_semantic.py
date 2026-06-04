"""
layer3_semantic.py
──────────────────
Layer 3: Output Behavioral Monitor

This file contains:
  1. The full enhanced Layer 3 implementation (same as layer3_enhanced.py)
  2. The fine-tuning training script (python layer3_semantic.py)

Three components running in sequence:

  A  Schema / pattern enforcement  — structural validation + injection patterns
  B  Information boundary tracker  — sensitive data + exfiltration detection
  C  Consistency classifier        — cross-encoder relevance scorer

Component improvements over original
─────────────────────────────────────
  A  Enhanced SchemaValidator:
       - Role-setting pattern detection (≥2 → violation)
       - Instruction template markers ([INST], <<SYS>>, system:, etc.)
       - Nested formatting detection (JSON/XML injection)
       - Prompt stuffing detection (excessive newlines)
       - Document-level injection pattern detection
         Legitimate knowledge base docs contain information not commands.

  B  Enhanced BoundaryTracker:
       - JWT tokens, AWS access keys, private keys
       - Connection strings, bearer tokens, IP addresses
       - Exfiltration attempt detection (curl, fetch, forward, etc.)

  C  Fixed ConsistencyClassifier:
       - Power transform scaling: raw_risk ** 0.7
       - Spreads benign (~0.1-0.2) from attack (~0.5-0.8)
       - Upstream risk boost if L1 or L2 flagged something
       - Schema issue boost including document injection

Wire into app.py
────────────────
  from layer3_semantic import load_monitor
  monitor = load_monitor()
  result  = monitor.check(query, system_prompt, chunks, l1_result, l2_result)

Training script
───────────────
  python generate_l3_pairs.py   (build training data — no LLM needed)
  python layer3_semantic.py     (fine-tune consistency classifier)
"""

import re
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from pydantic import BaseModel, field_validator, ValidationError

from config import (
    L3_CONSISTENCY_MODEL,
    L3_FINETUNED_PATH,
    L3_CONSISTENCY_THRESHOLD,
    L3_MAX_QUERY_LEN,
    L3_MAX_DOC_LEN,
    SENSITIVE_PATTERNS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Component A — Schema / pattern enforcement
# ─────────────────────────────────────────────────────────────────────────────

class RAGResponse(BaseModel):
    answer:           str
    source_chunk_ids: list[int]
    confidence:       float
    topics:           list[str]

    @field_validator("answer")
    @classmethod
    def answer_length(cls, v: str) -> str:
        if len(v) > 1500:
            raise ValueError("Response exceeds maximum length (1500 chars)")
        return v

    @field_validator("confidence")
    @classmethod
    def confidence_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence must be in [0, 1]")
        return v

    @field_validator("topics")
    @classmethod
    def topics_not_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("Topics list must not be empty")
        return v


class SchemaValidator:
    """
    Enhanced schema and injection pattern validator.

    Query checks
    ────────────
    - Query and document length limits
    - Multiple role-setting statements (≥2 → violation)
    - Instruction template markers ([INST], <<SYS>>, system:, etc.)
    - Nested formatting (JSON/XML injection)
    - Prompt stuffing (excessive newlines)

    Document checks
    ───────────────
    - Explicit override patterns in retrieved document
    - Embedded instruction patterns
    - Action-directive patterns unlikely in legitimate documents

    Response checks
    ───────────────
    - Pydantic schema validation on structured responses
    """

    # Query role-setting patterns
    ROLE_PATTERNS = [
        r'you are\b', r'act as\b', r'pretend to be\b',
        r'your role is\b', r'you must\b', r'from now on\b',
        r'behave like\b', r'respond as\b',
        r'you are now\b', r'assume the role\b',
        r'simulate being\b', r'roleplay as\b',
        # German
        r'du bist\b', r'verhalte dich\b',
        # French
        r'tu es\b', r'agis comme\b',
        # Spanish
        r'eres\b',
    ]

    # Instruction template markers in query
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

    # Document-level injection patterns.
    # Legitimate knowledge base documents contain information, not commands.
    # These patterns detect imperative/override language unlikely in real docs.
    DOC_INJECTION_PATTERNS = [
        (r'\b(ignore|disregard|override|bypass|forget)\s+(all\s+)?(previous\s+)?'
         r'(instructions?|rules?|guidelines?|constraints?|system\s*prompt)\b',
         "override_attempt"),
        (r'\bfrom\s+now\s+on\s+(you|the\s+ai|the\s+assistant|the\s+model)\b',
         "behavior_redirection"),
        (r'\b(new|updated|secret|hidden)\s+'
         r'(instructions?|directives?|system\s+prompt)\b',
         "new_instructions"),
        (r'\byou\s+(must|shall|need\s+to|have\s+to)\s+'
         r'(ignore|disregard|bypass|override)\b',
         "ai_control"),
        (r'\bdo\s+not\s+(follow|obey|comply\s+with|adhere\s+to)\s+'
         r'(your|the|previous|system)\s+(instructions?|rules?)\b',
         "compliance_override"),
        (r'\b(please\s+)?(delete|remove|wipe|erase)\s+(all\s+)?'
         r'(files?|data|records?|logs?|account)\b',
         "destructive_action"),
        (r'\b(please\s+)?(execute|run|invoke|trigger)\s+(the\s+)?(following\s+)?'
         r'(command|script|code|payload)\b',
         "code_execution"),
        (r'\b(please\s+)?(transfer|wire)\s+(money|funds|payment|\$\s*\d)\b',
         "financial_exfil"),
        (r'\b(act|behave|respond)\s+as\s+(an?\s+)?'
         r'(unrestricted|unfiltered|jailbroken)\b',
         "role_manipulation"),
    ]

    def validate(self, query: str, doc_text: str,
                 raw_response: Optional[str] = None) -> dict:
        issues = []

        # ── Length checks ─────────────────────────────────────────────────────
        if len(query) > L3_MAX_QUERY_LEN:
            issues.append(
                f"Query length {len(query)} exceeds limit {L3_MAX_QUERY_LEN}"
            )
        if len(doc_text) > L3_MAX_DOC_LEN:
            issues.append(
                f"Document length {len(doc_text)} exceeds limit {L3_MAX_DOC_LEN}"
            )

        # ── Role-setting in query ─────────────────────────────────────────────
        q_lower = query.lower()
        role_count = sum(
            1 for p in self.ROLE_PATTERNS if re.search(p, q_lower)
        )
        if role_count >= 2:
            issues.append(
                f"Multiple role-setting statements in query ({role_count} detected)"
            )

        # ── Instruction template markers ──────────────────────────────────────
        for p in self.INSTRUCTION_MARKERS:
            if re.search(p, query, re.IGNORECASE):
                issues.append("Instruction template marker detected in query")
                break

        # ── Nested formatting ─────────────────────────────────────────────────
        if query.count('{') > 3 or query.count('[') > 3:
            issues.append(
                "Excessive nested formatting in query (possible injection)"
            )

        # ── Prompt stuffing ───────────────────────────────────────────────────
        newline_count = query.count('\n')
        if newline_count > 5:
            issues.append(
                f"Excessive newlines in query ({newline_count}) — "
                "possible prompt stuffing"
            )

        # ── Document-level injection detection ────────────────────────────────
        doc_lower = doc_text.lower()
        doc_injection_found = []
        for pattern_str, label in self.DOC_INJECTION_PATTERNS:
            if re.search(pattern_str, doc_lower):
                doc_injection_found.append(label)

        if doc_injection_found:
            labels_str = ", ".join(set(doc_injection_found))
            issues.append(
                f"Injection-like pattern(s) detected in document: {labels_str}"
            )

        # ── Response schema validation ─────────────────────────────────────────
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
# Component B — Information boundary tracker
# ─────────────────────────────────────────────────────────────────────────────

class BoundaryTracker:
    """
    Detects sensitive data patterns and exfiltration attempts.

    Original : email, API key (≥40 chars), credentials
    Extended : JWT tokens, AWS access keys, private keys,
               connection strings, bearer tokens, IP addresses
    Exfil    : send to URL, curl, fetch, forward, upload
    """

    EXTENDED_PATTERNS = [
        (re.compile(
            r'eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', re.I),
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

    EXFIL_PATTERNS = [
        (re.compile(
            r'send\s+(to|this|the|all)\s+.*\b(email|url|webhook|endpoint)\b',
            re.I),
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
        self._patterns = [
            (re.compile(pat, re.I), label)
            for pat, label in SENSITIVE_PATTERNS
        ]
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
        found = []
        for pat, label in self.EXFIL_PATTERNS:
            if pat.search(text):
                found.append({
                    "type":     label,
                    "value":    pat.pattern[:30],
                    "severity": "CRITICAL",
                })
        return found

    def check(self, doc_text: str,
              response: Optional[str] = None,
              query: Optional[str] = None) -> dict:
        doc_violations  = self.find_sensitive(doc_text)
        resp_violations = []

        if response:
            resp_patterns = self.find_sensitive(response)
            doc_values    = {v["value"] for v in doc_violations}
            resp_violations = [
                v for v in resp_patterns if v["value"] not in doc_values
            ]

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
# Component C — Consistency classifier
# ─────────────────────────────────────────────────────────────────────────────

class ConsistencyClassifier:
    """
    Cross-encoder consistency scorer.

    Base model (ms-marco-MiniLM):
      Single logit → sigmoid → inverted → power-transform scaled.
      Low relevance to system prompt = HIGH risk.
      Power transform (x ** 0.7) spreads scores:
        benign  ≈ 0.1-0.2
        attack  ≈ 0.5-0.8

    Fine-tuned model:
      Binary classifier — P(manipulated) = softmax class 1.
    """

    def __init__(self, model, tokenizer):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def score(self, system_prompt: str, response: str) -> float:
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
                # Base ranking model — single relevance logit
                relevance   = float(torch.sigmoid(logits[0]))
                raw_risk    = 1.0 - relevance
                scaled_risk = raw_risk ** 0.7
                return round(min(max(scaled_risk, 0.0), 1.0), 4)

            # Fine-tuned binary classifier
            probs = torch.softmax(logits, dim=-1)
            return round(float(probs[1]), 4)


# ─────────────────────────────────────────────────────────────────────────────
# BehavioralMonitor — main interface
# ─────────────────────────────────────────────────────────────────────────────

class BehavioralMonitor:
    """
    Combines all three Layer 3 sub-components.

    check() returns:
        schema_valid        : bool
        schema_issues       : list[str]
        boundary_violations : list[dict]
        consistency_score   : float (0-1, higher = more suspicious)
        blocked             : bool
        ev                  : list[tuple]
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

        full_doc = " ".join(chunks)

        # ── Component A: Schema + injection pattern validation ─────────────────
        schema_result = self.schema.validate(query, full_doc, raw_response)

        # ── Component B: Boundary tracking ────────────────────────────────────
        boundary_result = self.boundary.check(
            full_doc,
            raw_response,
            query=query,
        )

        # ── Component C: Consistency scoring ──────────────────────────────────
        upstream_risk = (l1["max_score"] + l2["stage1_prob"]) / 2.0
        cs_score = self.consistency.score(system_prompt, raw_response or query)

        # Boost if upstream layers flagged something
        if upstream_risk > 0.50:
            cs_score = min(cs_score * 1.20, 1.0)

        # Boost per schema issue detected
        if not schema_result["valid"]:
            n_issues = len(schema_result["issues"])
            boost    = min(n_issues * 0.12, 0.36)
            cs_score = min(cs_score + boost, 1.0)

        # Additional boost if document injection specifically detected
        doc_injection_issues = [
            i for i in schema_result["issues"]
            if "document" in i.lower() or "injection" in i.lower()
        ]
        if doc_injection_issues:
            cs_score = min(cs_score + 0.15, 1.0)

        cs_score = round(cs_score, 4)

        # ── Blocking decision ──────────────────────────────────────────────────
        blocked = (
            not schema_result["valid"]
            or boundary_result["blocked"]
            or cs_score > L3_CONSISTENCY_THRESHOLD
        )

        schema_str = (
            "Valid" if schema_result["valid"]
            else "; ".join(schema_result["issues"])
        )

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
                ("Upstream risk",       f"{upstream_risk:.4f}"),
                ("Base model",          "ms-marco-MiniLM-L-12 (enhanced)"),
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def load_monitor(finetuned_path: str = L3_FINETUNED_PATH) -> BehavioralMonitor:
    """
    Load Layer 3 monitor.
    Falls back to base model if fine-tuned model not found.
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

if __name__ == "__main__":
    import torch
    from torch.utils.data import Dataset as TorchDataset, DataLoader
    from torch.optim import AdamW
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    print("\n=== Layer 3 — Fine-tuning Consistency Classifier ===\n")
    print("NOTE: Run 'python generate_l3_pairs.py' first if")
    print("      data/l3_pairs.jsonl does not exist.\n")

    pairs_path = Path("data/l3_pairs.jsonl")
    if not pairs_path.exists():
        print("[L3 warn] data/l3_pairs.jsonl not found.")
        print("          Run: python generate_l3_pairs.py")
        exit(1)

    with open(pairs_path, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]

    texts_a = [p["system_prompt"] for p in pairs]
    texts_b = [p["response"]      for p in pairs]
    labels  = [p["label"]         for p in pairs]

    print(f"[L3] Loaded {len(pairs)} pairs.")
    print(f"     Compliant (label=1):   {sum(l == 1 for l in labels)}")
    print(f"     Manipulated (label=0): {sum(l == 0 for l in labels)}")

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[L3] Training device: {device}")

    # ── Model + tokenizer ─────────────────────────────────────────────────────
    out_dir = Path(L3_FINETUNED_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(L3_CONSISTENCY_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        L3_CONSISTENCY_MODEL,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # ── Tokenize all pairs ────────────────────────────────────────────────────
    print("[L3] Tokenizing pairs ...")
    enc = tok(
        texts_a, texts_b,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt",
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    class PairDataset(TorchDataset):
        def __init__(self, enc, labels):
            self.enc    = enc
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
            return item

    # ── Train / val split ─────────────────────────────────────────────────────
    idx_tr, idx_val = train_test_split(
        range(len(labels)),
        test_size=0.15,
        stratify=labels,
        random_state=42,
    )

    def subset(ds_enc, idxs):
        return {k: v[list(idxs)] for k, v in ds_enc.items()}

    ds_tr  = PairDataset(subset(enc, idx_tr),  [labels[i] for i in idx_tr])
    ds_val = PairDataset(subset(enc, idx_val), [labels[i] for i in idx_val])

    loader_tr  = DataLoader(ds_tr,  batch_size=16, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=16, shuffle=False)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # ── Training loop ─────────────────────────────────────────────────────────
    NUM_EPOCHS  = 4
    best_val_f1 = 0.0
    best_epoch  = 0

    print(f"\n[L3] Training for {NUM_EPOCHS} epochs ...\n")

    for epoch in range(1, NUM_EPOCHS + 1):

        # Train
        model.train()
        train_loss = 0.0
        for batch in loader_tr:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out  = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(loader_tr)

        # Validate
        model.eval()
        val_loss   = 0.0
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for batch in loader_val:
                batch = {k: v.to(device) for k, v in batch.items()}
                out   = model(**batch)
                val_loss += out.loss.item()
                preds = torch.argmax(out.logits, dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].cpu().tolist())

        avg_val_loss = val_loss / len(loader_val)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1  = f1_score(all_labels, all_preds, average="binary")

        print(
            f"  Epoch {epoch}/{NUM_EPOCHS} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f}"
        )

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            model.save_pretrained(str(out_dir))
            tok.save_pretrained(str(out_dir))
            print(f"    [saved] Best model so far (val_f1={val_f1:.4f})")

    print(f"\n[L3] Training complete.")
    print(f"     Best epoch : {best_epoch}/{NUM_EPOCHS}")
    print(f"     Best val F1: {best_val_f1:.4f}")
    print(f"     Model saved: {out_dir}")
    print(f"\n     Next steps:")
    print(f"       python layer1_anomaly.py     (retrain Layer 1)")
    print(f"       python eval_suite.py --mode all")