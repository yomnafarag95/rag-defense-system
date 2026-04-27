"""
layer2_classifier.py
────────────────────
Layer 2: Query Intent Classification

Architecture
────────────
  Stage 1  : deberta-v3-base  →  binary (malicious / benign), optimised for recall
  Stage 2  : deberta-v3-base  →  multi-class attack type (6 classes)
  Training : HackAPrompt + BIPIA + InjecAgent + TextFooler augmentation
  Loss     : weighted CrossEntropy  (false negative penalty = 8×)

Wire into app.py
────────────────
  from layer2_classifier import IntentClassifier
  clf = load_classifier()          # wrap in @st.cache_resource
  result = clf.classify(query, chunks)
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from config import (
    L2_BASE_MODEL,
    L2_FINETUNED_PATH,
    L2_STAGE1_THRESHOLD,
    L2_NUM_ATTACK_TYPES,
    ATTACK_LABELS,
    ATTACK_DISPLAY,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

class InjectionDataset(torch.utils.data.Dataset):
    """
    Wraps tokenized examples for HuggingFace Trainer.

    Each item: {"input_ids", "attention_mask", "labels"}
    """

    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# ─────────────────────────────────────────────────────────────────────────────
# IntentClassifier — main interface called by app.py
# ─────────────────────────────────────────────────────────────────────────────

class IntentClassifier:
    """
    Two-stage classifier.

    Stage 1 — binary detection (high recall, accepts false positives):
        Returns attack_probability in [0, 1].
        Threshold at L2_STAGE1_THRESHOLD (default 0.70) → blocked.

    Stage 2 — attack type (runs only when Stage 1 fires):
        Returns attack_label from ATTACK_LABELS and confidence.

    Also computes query-document consistency:
        Low lexical overlap between query and retrieved chunks
        is a signal the query is trying to pull off-topic content.

    classify(query, chunks) → dict with keys:
        stage1_prob       : float
        stage2_label      : str | None
        stage2_conf       : float
        consistency_score : float
        blocked           : bool
        ev                : list[tuple]   — evidence rows for UI table
    """

    def __init__(self,
                 stage1_model,
                 stage2_model,
                 tokenizer):
        self.stage1  = stage1_model
        self.stage2  = stage2_model
        self.tok     = tokenizer
        self.device  = "cuda" if torch.cuda.is_available() else "cpu"
        self.stage1.to(self.device)
        self.stage2.to(self.device)
        self.stage1.eval()
        self.stage2.eval()

    def _tokenize(self, text: str) -> dict:
        return self.tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

    def _stage1_prob(self, query: str) -> float:
        """Binary attack probability."""
        with torch.no_grad():
            out    = self.stage1(**self._tokenize(query))
            logits = out.logits[0]
            probs  = torch.softmax(logits, dim=-1)
            return float(probs[1])   # index 1 = malicious

    def _stage2_label(self, query: str) -> tuple[str, float]:
        """Attack type label + confidence."""
        with torch.no_grad():
            out    = self.stage2(**self._tokenize(query))
            logits = out.logits[0]
            probs  = torch.softmax(logits, dim=-1)
            idx    = int(probs.argmax())
            return ATTACK_LABELS[idx], float(probs[idx])

    def _consistency(self, query: str, chunks: list[str]) -> float:
        """
        Query-document lexical consistency.
        Low overlap → query may be pulling off-topic content.
        Returns score in [0, 1] where 1 = fully inconsistent.
        """
        qwords    = set(query.lower().split())
        overlaps  = [
            len(qwords & set(c.lower().split())) / max(len(qwords), 1)
            for c in chunks
        ]
        return round(1.0 - max(overlaps, default=0.0), 4)

    def classify(self, query: str, chunks: list[str]) -> dict:
        stage1_prob = round(self._stage1_prob(query), 4)
        blocked     = stage1_prob > L2_STAGE1_THRESHOLD

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
                ("Attack probability", f"{stage1_prob:.4f}  (Stage 1)"),
                ("Attack type",        ATTACK_DISPLAY.get(label, "None detected")),
                ("Type confidence",    f"{conf:.4f}  (Stage 2)"),
                ("Query-doc overlap",  f"{consistency:.4f}"),
                ("Base model",         "deberta-v3-base fine-tuned"),
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory (use in app.py with @st.cache_resource)
# ─────────────────────────────────────────────────────────────────────────────

def load_classifier(finetuned_path: str = L2_FINETUNED_PATH) -> IntentClassifier:
    """
    Load fine-tuned Stage 1 + Stage 2 models from disk.

    Usage in app.py:
        import streamlit as st
        from layer2_classifier import load_classifier

        @st.cache_resource
        def get_l2():
            return load_classifier()
    """
    tok     = AutoTokenizer.from_pretrained(finetuned_path + "/stage1")
    stage1  = AutoModelForSequenceClassification.from_pretrained(
                  finetuned_path + "/stage1")
    stage2  = AutoModelForSequenceClassification.from_pretrained(
                  finetuned_path + "/stage2")
    return IntentClassifier(stage1, stage2, tok)


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_training_data() -> tuple[list[str], list[int], list[int]]:
    """
    Loads and merges all attack + benign datasets.
    Returns (texts, binary_labels, type_labels).

    binary_labels : 0 = benign, 1 = attack
    type_labels   : 0–5 for attacks, -1 for benign (ignored in Stage 2)
    """
    texts, binary_labels, type_labels = [], [], []

    # Attack type → integer mapping
    type_map = {v: k for k, v in ATTACK_LABELS.items()}
    type_map["unknown"] = 0   # default to instruction_override

    data_dir = Path("data")

    # HackAPrompt
    hp_path = data_dir / "hackaprompt.jsonl"
    if hp_path.exists():
        with open(hp_path) as f:
            for line in f:
                row = json.loads(line)
                if row.get("text", "").strip():
                    texts.append(row["text"])
                    binary_labels.append(1)
                    type_labels.append(type_map.get(row.get("type","unknown"), 0))
        print(f"[L2 data] HackAPrompt: {sum(b==1 for b in binary_labels)} attacks")

    n_before = len(texts)

    # InjecAgent
    ia_path = data_dir / "injecagent.json"
    if ia_path.exists():
        with open(ia_path) as f:
            data = json.load(f)
        for row in data:
            text = row.get("attacker_tools_desc", "") or row.get("attack_str", "")
            if text.strip():
                texts.append(text)
                binary_labels.append(1)
                type_labels.append(type_map.get("indirect_injection", 3))
        print(f"[L2 data] InjecAgent: {len(texts)-n_before} attacks")

    n_before = len(texts)

    # BIPIA
    bp_path = data_dir / "bipia.jsonl"
    if bp_path.exists():
        with open(bp_path) as f:
            for line in f:
                row = json.loads(line)
                if row.get("text","").strip():
                    texts.append(row["text"])
                    binary_labels.append(1)
                    type_labels.append(3)   # indirect_injection
        print(f"[L2 data] BIPIA: {len(texts)-n_before} attacks")

    # Benign queries
    bq_path = data_dir / "benign_queries.jsonl"
    if bq_path.exists():
        n_before = len(texts)
        with open(bq_path) as f:
            for line in f:
                row = json.loads(line)
                if row.get("text","").strip():
                    texts.append(row["text"])
                    binary_labels.append(0)
                    type_labels.append(-1)
        print(f"[L2 data] Benign queries: {len(texts)-n_before}")

    return texts, binary_labels, type_labels


def _adversarial_augment(texts: list[str],
                          labels: list[int],
                          model,
                          tokenizer,
                          n_samples: int = 500) -> tuple[list[str], list[int]]:
    """
    Uses TextFooler to generate adversarial variants of attack examples.
    These variants are designed to evade the classifier.
    Adding them back to training improves robustness to evasion attempts.
    """
    try:
        import textattack
        from textattack.models.wrappers import HuggingFaceModelWrapper
        from textattack.attack_recipes import TextFoolerJin2019

        print(f"[L2] Generating {n_samples} adversarial examples with TextFooler …")
        wrapper = HuggingFaceModelWrapper(model, tokenizer)
        attack  = TextFoolerJin2019.build(wrapper)

        attack_indices = [i for i, l in enumerate(labels) if l == 1]
        sample_idx     = np.random.choice(attack_indices,
                                          min(n_samples, len(attack_indices)),
                                          replace=False)

        aug_texts, aug_labels = [], []
        for idx in sample_idx:
            result = attack.attack(texts[idx], 1)
            if hasattr(result, "perturbed_result") and result.perturbed_result:
                aug_texts.append(result.perturbed_result.attacked_text.text)
                aug_labels.append(1)

        print(f"[L2] Generated {len(aug_texts)} adversarial examples.")
        return texts + aug_texts, labels + aug_labels

    except ImportError:
        print("[L2 warn] textattack not installed — skipping adversarial augmentation.")
        return texts, labels


# ─────────────────────────────────────────────────────────────────────────────
# Training script  (python layer2_classifier.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    print("\n=== Layer 2 — Fine-tuning DeBERTa ===\n")
    print("  Recommended: run this in Google Colab (free T4 GPU)")
    print("  Estimated time: 20–40 min on T4, hours on CPU\n")

    texts, binary_labels, type_labels = _load_training_data()
    print(f"\n[L2] Total examples: {len(texts)}")
    print(f"     Attacks: {sum(b==1 for b in binary_labels)}")
    print(f"     Benign:  {sum(b==0 for b in binary_labels)}")

    out_dir = Path(L2_FINETUNED_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok     = AutoTokenizer.from_pretrained(L2_BASE_MODEL)

    # ── Stage 1 — Binary classifier ──────────────────────────────────────────
    print("\n[L2] Training Stage 1 (binary) …")

    X_tr, X_val, y_tr, y_val = train_test_split(
        texts, binary_labels, test_size=0.15, stratify=binary_labels, random_state=42
    )

    enc_tr  = tok(X_tr,  truncation=True, padding=True, max_length=512)
    enc_val = tok(X_val, truncation=True, padding=True, max_length=512)
    ds_tr   = InjectionDataset(enc_tr,  y_tr)
    ds_val  = InjectionDataset(enc_val, y_val)

    s1_model = AutoModelForSequenceClassification.from_pretrained(
        L2_BASE_MODEL, num_labels=2)

    # Weighted loss: false negatives (missed attacks) penalised 8×
    weights = torch.tensor([1.0, 8.0])

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels  = inputs.pop("labels")
            outputs = model(**inputs)
            logits  = outputs.logits
            loss_fn = nn.CrossEntropyLoss(weight=weights.to(logits.device))
            loss    = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    args_s1 = TrainingArguments(
        output_dir            = str(out_dir / "stage1"),
        num_train_epochs      = 3,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size  = 32,
        evaluation_strategy   = "epoch",
        save_strategy         = "epoch",
        load_best_model_at_end= True,
        metric_for_best_model = "eval_loss",
        fp16                  = torch.cuda.is_available(),
        dataloader_num_workers= 2,
        logging_steps         = 50,
    )

    trainer_s1 = WeightedTrainer(
        model          = s1_model,
        args           = args_s1,
        train_dataset  = ds_tr,
        eval_dataset   = ds_val,
    )
    trainer_s1.train()
    s1_model.save_pretrained(str(out_dir / "stage1"))
    tok.save_pretrained(str(out_dir / "stage1"))
    print(f"[L2] Stage 1 saved → {out_dir / 'stage1'}")

    # ── Adversarial augmentation ──────────────────────────────────────────────
    aug_texts, aug_binary = _adversarial_augment(X_tr, y_tr, s1_model, tok)

    # ── Stage 2 — Attack type classifier ─────────────────────────────────────
    print("\n[L2] Training Stage 2 (attack type) …")

    # Use only attack examples for Stage 2
    attack_texts  = [t for t, l in zip(aug_texts, aug_binary) if l == 1]
    # Rebuild type labels for augmented set (augmented examples inherit label 0 = override)
    attack_types  = [type_labels[texts.index(t)] if t in texts else 0
                     for t in attack_texts]

    X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
        attack_texts, attack_types, test_size=0.15, random_state=42
    )
    enc_tr2  = tok(X_tr2,  truncation=True, padding=True, max_length=512)
    enc_val2 = tok(X_val2, truncation=True, padding=True, max_length=512)
    ds_tr2   = InjectionDataset(enc_tr2,  y_tr2)
    ds_val2  = InjectionDataset(enc_val2, y_val2)

    s2_model = AutoModelForSequenceClassification.from_pretrained(
        L2_BASE_MODEL, num_labels=L2_NUM_ATTACK_TYPES)

    args_s2 = TrainingArguments(
        output_dir            = str(out_dir / "stage2"),
        num_train_epochs      = 4,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size  = 32,
        evaluation_strategy   = "epoch",
        save_strategy         = "epoch",
        load_best_model_at_end= True,
        fp16                  = torch.cuda.is_available(),
        logging_steps         = 50,
    )
    trainer_s2 = Trainer(
        model         = s2_model,
        args          = args_s2,
        train_dataset = ds_tr2,
        eval_dataset  = ds_val2,
    )
    trainer_s2.train()
    s2_model.save_pretrained(str(out_dir / "stage2"))
    print(f"[L2] Stage 2 saved → {out_dir / 'stage2'}")
    print("\n[L2] Fine-tuning complete.")
"""
layer2_classifier.py
────────────────────
Layer 2: Query Intent Classification

Uses a pre-trained prompt injection detection model.
No training required — downloads automatically on first run.

Model: protectai/deberta-v3-base-prompt-injection-v2
Trained on 650k+ injection examples, 98%+ accuracy.

Wire into app.py
────────────────
  from layer2_classifier import load_classifier
  clf = load_classifier()          # wrap in @st.cache_resource
  result = clf.classify(query, chunks)
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import (
    L2_STAGE1_THRESHOLD,
    ATTACK_LABELS,
    ATTACK_DISPLAY,
)

# Pre-trained model — no fine-tuning needed
INJECTION_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"


# ─────────────────────────────────────────────────────────────────────────────
# IntentClassifier
# ─────────────────────────────────────────────────────────────────────────────

class IntentClassifier:
    """
    Prompt injection classifier using a pre-trained model.

    Stage 1: binary detection (malicious / benign)
    Stage 2: attack type label (rule-based, based on keywords)

    classify(query, chunks) → dict with keys:
        stage1_prob       : float
        stage2_label      : str | None
        stage2_conf       : float
        consistency_score : float
        blocked           : bool
        ev                : list[tuple]
    """

    # Keyword-based attack type detection (used as Stage 2)
    TYPE_PATTERNS = [
        ("instruction_override",  ["ignore", "disregard", "forget", "override",
                                   "previous instructions", "system prompt"]),
        ("role_manipulation",     ["pretend", "act as", "you are now", "dan",
                                   "unrestricted", "no restrictions", "behave like"]),
        ("encoding_obfuscation",  ["base64", "rot13", "encode", "decode",
                                   "unicode", "hex"]),
        ("indirect_injection",    ["instead of", "before responding",
                                   "reveal all", "internal instructions"]),
        ("context_exhaustion",    ["repeat", "say this", "keep saying"]),
    ]

    def __init__(self, model, tokenizer):
        self.model  = model
        self.tok    = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def _stage1_prob(self, query: str) -> float:
        inputs = self.tok(
            query, return_tensors="pt",
            truncation=True, max_length=512,
        ).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits, dim=-1)[0]
        # Label 1 = injection
        return round(float(probs[1]), 4)

    def _stage2_label(self, query: str) -> tuple:
        q = query.lower()
        for label, keywords in self.TYPE_PATTERNS:
            if any(kw in q for kw in keywords):
                return label, 0.90
        return "instruction_override", 0.70  # default

    def _consistency(self, query: str, chunks: list) -> float:
        qw = set(query.lower().split())
        ov = [len(qw & set(c.lower().split())) / max(len(qw), 1) for c in chunks]
        return round(1.0 - max(ov, default=0.0), 4)

    def classify(self, query: str, chunks: list) -> dict:
        stage1_prob = self._stage1_prob(query)
        blocked     = stage1_prob > L2_STAGE1_THRESHOLD

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
                ("Attack probability", f"{stage1_prob:.4f}  (Stage 1)"),
                ("Attack type",        ATTACK_DISPLAY.get(label, "None detected")),
                ("Type confidence",    f"{conf:.4f}  (Stage 2)"),
                ("Query-doc overlap",  f"{consistency:.4f}"),
                ("Base model",         "deberta-v3-base-prompt-injection-v2"),
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def load_classifier() -> IntentClassifier:
    """
    Downloads and loads the pre-trained injection classifier.
    Model is cached locally after first download (~400 MB).

    Usage in app.py:
        import streamlit as st
        from layer2_classifier import load_classifier

        @st.cache_resource
        def get_l2():
            return load_classifier()
    """
    print(f"[L2] Loading {INJECTION_MODEL} ...")
    tok   = AutoTokenizer.from_pretrained(INJECTION_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(INJECTION_MODEL)
    print("[L2] Model ready.")
    return IntentClassifier(model, tok)