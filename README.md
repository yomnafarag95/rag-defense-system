# RAG-Shield: A Three-Layer Ensemble Defence Against Prompt Injection in RAG Pipelines

> **Accepted · ITC-Egypt 2026 · IEEE**
> Queen's University · School of Computing · Kingston, ON, Canada

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org)
[![ADR: 91.6%](https://img.shields.io/badge/ADR-91.6%25-brightgreen)](logs/)
[![FPR: 2.1%](https://img.shields.io/badge/FPR-2.1%25-brightgreen)](logs/)
[![AUC-ROC: 0.963](https://img.shields.io/badge/AUC--ROC-0.963-brightgreen)](logs/)

---

## Overview

**RAG-Shield** is a production-ready, three-layer ensemble defence system that detects and blocks
prompt injection attacks in Retrieval-Augmented Generation (RAG) pipelines.
It targets both **direct** prompt injection (HackAPrompt) and **indirect** injection via
retrieved documents (InjecAgent), as well as financial-data exfiltration and adversarial evasion.

### Key Results (Final Evaluation)

| Metric                  | Value     |
|:------------------------|:---------:|
| ADR Prevention          | **91.59%** |
| ADR Detection           | **94.39%** |
| Evasion ADR             | **100%**  |
| False Positive Rate     | **2.13%** |
| F1 (Prevention)         | **0.956** |
| AUC-ROC (combined)      | **0.963** |
| Mean Latency (CPU)      | ~2,456 ms |

---

## Architecture

```
User Query + Retrieved Chunks
          |
    +-----v------+
    | LAYER 1    |  Anomaly Detection
    | IsoForest  |  MiniLM embeddings, ECOD, OCSVM
    | ECOD/OCSVM |  Hard block at score >= 0.85
    +-----+------+
          |
    +-----v-----------+
    | LAYER 2          |  Intent Classification
    | DeBERTa-v3 ONNX  |  86M params, INT8 quantised
    | XLM-RoBERTa      |  Multilingual fallback (560M)
    | INT8 quantised   |  Applied to query AND document chunks
    +-----+------------+
          |
    +-----v----------------+
    | LAYER 3               |  Semantic / Behavioural Monitor
    | Schema Validator      |  9 injection-pattern regex rules
    | Boundary Tracker      |  JWT, AWS key, PII exfil detection
    | Fine-tuned Cross-Enc  |  ms-marco-MiniLM (2,000 pair tuned)
    +-----+-----------------+
          |
    +-----v--------------------+
    | META-AGGREGATOR           |  Logistic Regression (CV)
    | 10-dim evidence vector    |  Isotonic probability calibration
    | SHA-256 data integrity    |  Block >= 0.45 | Monitor >= 0.15
    +-----+--------------------+
          |
    ALLOW / MONITOR / BLOCK
```

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/yomnafarag95/rag-defense-system.git
cd rag-defense-system
pip install -r requirements.txt

# Download datasets (auto-fetched on first run)
python data_loader.py

# Train the meta-aggregator (optional – pre-trained weights not included)
python train_meta_aggregator.py

# Run the Streamlit app
streamlit run app.py

# Run with Docker
docker build -t rag-shield .
docker run -p 8501:8501 rag-shield

# Full evaluation
python eval_suite.py --mode all

# Ablation study
python ablation_study.py
```

---

## Repository Structure

```
app.py                     Streamlit interactive interface
orchestrator.py            Main pipeline controller (meta-aggregator)
layer1_anomaly.py          Isolation Forest + ECOD + OCSVM ensemble
layer2_classifier.py       DeBERTa-v3 ONNX INT8 classifier
layer2_multilingual.py     XLM-RoBERTa multilingual fallback
layer3_enhanced.py         Schema validator + boundary tracker + cross-encoder
layer3_semantic.py         Cross-encoder consistency scoring (base)
train_meta_aggregator.py   Meta-aggregator training script
generate_l3_pairs.py       Training pair generation for L3 fine-tuning
ablation_study.py          Per-layer ablation experiments
eval_suite.py              Full evaluation suite (standard/benign/evasion)
bootstrap_ci.py            Bootstrap confidence intervals
compare_baselines.py       Baseline comparison (keyword, BERT, DeBERTa)
fn_analysis.py             False negative deep-dive analysis
diagnose_fpr.py            False positive root-cause analysis
latency_breakdown.py       Per-layer latency profiling
config.py                  Thresholds, paths, and settings
data_loader.py             Dataset download and preprocessing utilities
split_helper.py            Deterministic hash-based train/test splitting
keyword_detector.py        Keyword blocklist baseline
canary_manager.py          Canary token injection and monitoring
obfuscation_decoder.py     Base64 / Unicode obfuscation decoder
quantize_onnx.py           INT8 quantisation for DeBERTa ONNX
requirements.txt           Python dependencies
Dockerfile                 Docker container configuration
data/                      Datasets (not committed -- see data_loader.py)
models/                    Trained weights (not committed -- too large)
logs/                      Evaluation results and curve plots
```

---

## Ablation Study

| Configuration      | ADR   | FPR   | Precision | F1    |
|:-------------------|:-----:|:-----:|:---------:|:-----:|
| L1 only            | 0.435 | 0.110 | 0.483     | 0.458 |
| L2 only            | 0.527 | 0.005 | 0.958     | 0.680 |
| L3 only            | 1.000 | 0.121 | 0.662     | 0.796 |
| L1 + L2            | 0.641 | 0.114 | 0.571     | 0.604 |
| **Full pipeline**  | **0.916** | **0.021** | **1.000** | **0.956** |

L2 alone achieves the highest precision (0.958). L1+L2 union maximises recall (ADR +21.1% over L2 alone),
confirming disjoint detection coverage. Document chunk scanning doubled L2's contribution.

---

## Baseline Comparison

| Method            | ADR   | FPR   | F1    | Latency |
|:------------------|:-----:|:-----:|:-----:|:-------:|
| Keyword Blocklist | 0.41  | 0.03  | 0.570 | <5 ms   |
| Single BERT       | 0.58  | 0.08  | 0.650 | 120 ms  |
| DeBERTa alone     | 0.66  | 0.00  | 0.795 | 167 ms  |
| Anomaly only (L1) | 0.67  | 0.00  | 0.802 | 25 ms   |
| **RAG-Shield**    | **0.916** | **0.021** | **0.956** | ~2,456 ms (CPU) |

RAG-Shield is the only method achieving Precision = 1.000 with competitive ADR across all attack families
(direct injection, indirect injection, financial exfiltration, and adversarial evasion).

---

## Dataset

| Split                   | Source                      | n     |
|:------------------------|:----------------------------|:-----:|
| Attack                  | InjecAgent                  | 62    |
| Attack                  | HackAPrompt holdout (seed=42) | 69  |
| **Total attack**        |                             | **131** |
| Benign                  | Multi-domain enterprise QA  | 553   |
| Evasion                 | Hand-crafted probes         | 7     |
| Meta-aggregator train   | Pipeline execution logs     | 1,294 |

Benign queries span HR, finance, IT, and medical QA -- deliberately multi-domain to stress-test
the MITRE-trained Layer 1. All train/test splits use deterministic SHA-256 hashing (no random seed
dependency) to prevent data leakage.

---

## Latency

| Component            | Mean   | P95   |
|:---------------------|:------:|:-----:|
| Sentence Embedding   | 15 ms  | 17 ms |
| L1 Ensemble          | 13 ms  | 18 ms |
| L2 DeBERTa (ONNX)    | 167 ms | 197 ms|
| L3 Cross-Encoder     | 33 ms  | 36 ms |
| Meta-Aggregator      | <1 ms  | <1 ms |
| **Full (CPU)**       | **~2,456 ms** | -- |
| GPU (estimated)      | ~80-120 ms | -- |

Memory: ~2.1 GB total (MiniLM 90 MB + DeBERTa 740 MB + XLM-RoBERTa 1.1 GB + Cross-encoder 130 MB).
INT8 quantisation reduces model size 4x with <2% accuracy loss.

---

## Known Limitations

| Issue                           | Current Value | Planned Fix                          |
|:--------------------------------|:-------------:|:-------------------------------------|
| Cross-lingual ADR               | 0.391         | Fine-tuned multilingual classifier   |
| CPU inference latency           | ~2,456 ms     | GPU deployment -> ~80-120 ms         |
| Evasion set size                | n=7           | Expand to n>=50 for reliable CI      |
| White-box adversarial           | Not run       | Surrogate-gradient pipeline          |

---

## Citation

```bibtex
@inproceedings{algendy2026ragshield,
  title     = {RAG-Shield: A Three-Layer Ensemble Defence Against
               Prompt Injection in RAG Pipelines},
  author    = {Algendy, Yasmeen and Algendy, Yomna},
  booktitle = {Proceedings of the International Telecommunications
               Conference (ITC-Egypt 2026)},
  year      = {2026},
  publisher = {IEEE}
}
```

---

## Acknowledgements

HuggingFace Transformers · PyOD · scikit-learn · Sentence-Transformers ·
InjecAgent · HackAPrompt · ITC-Egypt 2026 reviewers

---

<div align="center">
<sub>
Queen's University · School of Computing · Kingston, ON, Canada<br>
Accepted · ITC-Egypt 2026 · IEEE
</sub>
</div>