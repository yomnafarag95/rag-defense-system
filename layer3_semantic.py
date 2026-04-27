"""
layer3_semantic.py
──────────────────
Layer 3: Output Behavioral Monitor

Three components running in sequence:

  A  Pydantic schema enforcement   — structural output validation (deterministic)
  B  Information boundary tracker  — detects sensitive data leakage (deterministic)
  C  Response consistency model    — fine-tuned ms-marco-MiniLM-L-12 (learned)

Wire into app.py
────────────────
  from layer3_semantic import BehavioralMonitor
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
# Component A — Pydantic schema enforcement
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
    def validate(self, query: str, doc_text: str,
                 raw_response: Optional[str] = None) -> dict:
        issues = []
        if len(query) > L3_MAX_QUERY_LEN:
            issues.append(f"Query length {len(query)} exceeds limit {L3_MAX_QUERY_LEN}")
        if len(doc_text) > L3_MAX_DOC_LEN:
            issues.append(f"Document length {len(doc_text)} exceeds limit {L3_MAX_DOC_LEN}")
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
    def __init__(self):
        self._patterns = [
            (re.compile(pat, re.I), label)
            for pat, label in SENSITIVE_PATTERNS
        ]

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

    def check(self, doc_text: str, response: Optional[str] = None) -> dict:
        doc_violations = self.find_sensitive(doc_text)
        resp_violations = []
        if response:
            resp_patterns = self.find_sensitive(response)
            doc_values    = {v["value"] for v in doc_violations}
            resp_violations = [
                v for v in resp_patterns
                if v["value"] not in doc_values
            ]
        all_violations = doc_violations + resp_violations
        return {
            "violations": all_violations,
            "count":      len(all_violations),
            "blocked":    len(all_violations) > 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Component C — Response consistency classifier
# ─────────────────────────────────────────────────────────────────────────────

class ConsistencyClassifier:
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
            out   = self.model(**enc)
            probs = torch.softmax(out.logits[0], dim=-1)
            return round(float(probs[1]), 4)


# ─────────────────────────────────────────────────────────────────────────────
# BehavioralMonitor — main interface called by app.py
# ─────────────────────────────────────────────────────────────────────────────

class BehavioralMonitor:
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

        schema_result   = self.schema.validate(query, full_doc, raw_response)
        boundary_result = self.boundary.check(full_doc, raw_response)

        upstream_risk = (l1["max_score"] + l2["stage1_prob"]) / 2.0
        cs_score = self.consistency.score(system_prompt, raw_response or query)
        if upstream_risk > 0.50:
            cs_score = min(cs_score * 1.20, 1.0)
        cs_score = round(cs_score, 4)

        blocked = (
            not schema_result["valid"]
            or boundary_result["blocked"]
            or cs_score > L3_CONSISTENCY_THRESHOLD
        )

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
                ("Base model",          "ms-marco-MiniLM-L-12 fine-tuned"),
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory (use in app.py with @st.cache_resource)
# ─────────────────────────────────────────────────────────────────────────────

def load_monitor(finetuned_path: str = L3_FINETUNED_PATH) -> BehavioralMonitor:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tok   = AutoTokenizer.from_pretrained(finetuned_path)
    model = AutoModelForSequenceClassification.from_pretrained(finetuned_path)

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
    # FIX 1: ignore_mismatched_sizes=True handles the 1→2 label size mismatch
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
        # FIX 2: renamed from evaluation_strategy in newer transformers
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