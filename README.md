---
title: RAG-Shield
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
app_file: src/streamlit_app.py
pinned: false
---

<div align="center">

# RAG-Shield
### Three-Layer Ensemble Defence Against Prompt Injection in RAG Pipelines

*Queen's University Â· School of Computing Â· ITC-Egypt 2026 Â· IEEE*

[![Paper](https://img.shields.io/badge/IEEE-ITC--Egypt%202026-2b6cb0?style=for-the-badge)](YOUR_PAPER_LINK_HERE)
[![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed?style=for-the-badge&logo=docker&logoColor=white)](Dockerfile)
[![HuggingFace](https://img.shields.io/badge/Models-HuggingFace-ff9d00?style=for-the-badge)](https://huggingface.co)

</div>

---

## What is RAG-Shield?

RAG systems expose a critical attack surface: adversaries can embed malicious instructions in queries or retrieved documents to hijack LLM behaviour. RAG-Shield defends against this with three independent detection layers fused by a calibrated meta-aggregator.

---

## Results

| Metric | Value |
|:---|:---:|
| ADR â€” 131 standard attacks | **67.2%** |
| ADR â€” 7 evasion probes | **85.7%** |
| Precision | **1.000** |
| F1 Score | **0.804** |
| FPR â€” 553 benign queries | **0.121** |
| AUC-ROC | **0.871** [0.841, 0.898] |
| Base64 pre-processing block rate | **62.7%** |

---

## Architecture

```
Query + Retrieved Documents + System Prompt
                    â”‚
                    â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  LAYER 1 Â· Unsupervised Anomaly         â”‚
â”‚  Isolation Forest + ECOD + One-Class SVMâ”‚
â”‚  Trained on MITRE ATT&CK (no labels).  â”‚
â”‚  Hard block at score â‰¥ 0.68.           â”‚
â”‚  Early exit ~25 ms Â· Detects 64.8% TP  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                   â”‚
                   â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  LAYER 2 Â· Multilingual Classifier      â”‚
â”‚  DeBERTa-v3 (86M) + XLM-RoBERTa (560M) â”‚
â”‚  Applied to query AND document chunks.  â”‚
â”‚  62.7% Base64 block at pre-processing.  â”‚
â”‚  Hard block at score â‰¥ 0.60.           â”‚
â”‚  Detects 30.7% TP Â· 66.7% evasion TP   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                   â”‚
                   â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  LAYER 3 Â· Semantic Monitor             â”‚
â”‚  Schema Validator Â· 9 regex patterns    â”‚
â”‚  Boundary Tracker Â· JWT, AWS, exfil     â”‚
â”‚  Cross-Encoder Â· consistency scoring    â”‚
â”‚  Hard block at boundary violations â‰¥ 2  â”‚
â”‚  Detects 4.5% TP Â· 16.7% evasion TP    â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                   â”‚
                   â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  META-AGGREGATOR Â· Logistic Regression  â”‚
â”‚  10-dim features Â· isotonic calibration â”‚
â”‚  Trained on 1,294 logs Â· SHA-256 verifiedâ”‚
â”‚  Block Î¸ = 0.45 Â· Monitor Î¸ = 0.15     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                   â”‚
        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
        â–¼          â–¼          â–¼
      ALLOW     MONITOR     BLOCK
```

---

## Ablation

| Config | ADR | FPR | Prec | F1 |
|:---|:---:|:---:|:---:|:---:|
| L1 only | 0.435 | 0.110 | 0.483 | 0.458 |
| L2 only | 0.527 | 0.005 | 0.958 | 0.680 |
| L3 only | 1.000 | 0.121 | 0.662 | 0.796 |
| L1 + L2 | 0.641 | 0.114 | 0.571 | 0.604 |
| **Full pipeline** | **0.672** | **0.121** | **1.000** | **0.804** |

> L2 alone achieves the highest precision (0.958). L1+L2 union maximises recall (ADR = 0.641, +21.1% over L2 alone), confirming disjoint detection coverage. Document chunk scanning doubled L2's contribution over query-only classification.
>
> **Note:** The ablation Full row (ADR = 0.580, meta-aggregator only) differs from the main result (ADR = 0.672) because per-layer hard-block rules fire before the meta-aggregator and are excluded from ablation configurations by design.

---

## Baselines

| Method | ADR | FPR | F1 | Latency |
|:---|:---:|:---:|:---:|:---:|
| Keyword Blocklist | 0.41 | 0.03 | 0.570 | < 5 ms |
| Single BERT | 0.58 | 0.08 | 0.650 | 120 ms |
| DeBERTa alone | 0.66 | 0.00 | 0.795 | 167 ms |
| Anomaly only (L1) | 0.67 | 0.00 | 0.802 | 25 ms |
| Monitor only (L3) | 1.00 | 1.00 | 0.490 | 33 ms |
| **RAG-Shield** | **0.672** | **0.121** | **0.804** | **469 ms** |

RAG-Shield is the only method achieving Precision = 1.000 with competitive ADR across all attack families.

---

## Latency

| Component | Mean | P95 |
|:---|:---:|:---:|
| Sentence Embedding | 15 ms | 17 ms |
| L1 Ensemble | 13 ms | 18 ms |
| L2 DeBERTa | 167 ms | 197 ms |
| L3 Cross-Encoder | 33 ms | 36 ms |
| Meta-Aggregator | < 1 ms | < 1 ms |
| **L1 early exit (clear attacks)** | **25 ms** | **33 ms** |
| **Full pipeline (CPU)** | **469 ms** | â€” |
| GPU (estimated) | ~80â€“120 ms | â€” |

Memory: ~2.1 GB total (MiniLM 90 MB Â· DeBERTa 740 MB Â· XLM-RoBERTa 1.1 GB Â· Cross-encoder 130 MB). INT8 quantisation reduces size 4Ã— with < 2% accuracy loss.

---

## Dataset

| Split | Source | n |
|:---|:---|:---:|
| Attack | InjecAgent | 62 |
| Attack | HackAPrompt holdout (seed=42) | 69 |
| **Total attack** | | **131** |
| Benign | Multi-domain enterprise QA | 553 |
| Evasion | Hand-crafted probes | 7 |
| Meta-aggregator train | Pipeline logs | 1,294 |

Benign queries span HR, finance, IT, code QA, and medical QA â€” deliberately multi-domain to stress-test the MITRE-trained Layer 1.

---

## Known Limitations

| Issue | Current Value | Plan |
|:---|:---:|:---|
| Benign FPR on multi-domain queries | 0.121 | Domain-specific L1 retraining â†’ target 0.050 |
| Cross-lingual ADR | 0.391 | Fine-tuned multilingual classifier (DE/FR/ZH/AR) |
| Evasion set size | n = 7 | Expand to n â‰¥ 50 for reliable CI |
| CPU latency with I/O | ~2,762 ms | GPU deployment â†’ ~80â€“120 ms |
| White-box adversarial evaluation | Not run | Surrogate-gradient pipeline (non-differentiable L1) |

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/yomnafarag95/rag-defense-system.git
cd rag-defense-system
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Run with Docker
docker build -t rag-shield .
docker run -p 8501:8501 rag-shield

# Run evaluation
python eval_suite.py

# Run ablation
python ablation_study.py
```

---

## Repository Structure

```
â”œâ”€â”€ app.py                     Streamlit interface
â”œâ”€â”€ orchestrator.py            Main pipeline controller
â”œâ”€â”€ layer1_anomaly.py          Isolation Forest + ECOD + OCSVM
â”œâ”€â”€ layer2_classifier.py       DeBERTa-v3 classifier
â”œâ”€â”€ layer2_multilingual.py     XLM-RoBERTa fallback
â”œâ”€â”€ layer3_enhanced.py         Schema + boundary + cross-encoder
â”œâ”€â”€ layer3_semantic.py         Cross-encoder scoring
â”œâ”€â”€ train_meta_aggregator.py   Meta-aggregator training
â”œâ”€â”€ ablation_study.py          Per-layer ablation
â”œâ”€â”€ eval_suite.py              Full evaluation pipeline
â”œâ”€â”€ bootstrap_ci.py            Bootstrap confidence intervals
â”œâ”€â”€ fn_analysis.py             False negative analysis
â”œâ”€â”€ latency_breakdown.py       Latency profiling
â”œâ”€â”€ config.py                  Thresholds and settings
â”œâ”€â”€ data_loader.py             Dataset utilities
â”œâ”€â”€ keyword_detector.py        Baseline blocklist
â”œâ”€â”€ requirements.txt           Dependencies
â”œâ”€â”€ Dockerfile                 Docker config
â””â”€â”€ figures/                   Paper figures
```

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

HuggingFace Transformers Â· PyOD Â· scikit-learn Â· Sentence-Transformers Â· InjecAgent Â· HackAPrompt Â· ITC-Egypt 2026 reviewers

---

<div align="center">
<sub>
Queen's University Â· School of Computing Â· Kingston, ON, Canada<br>
Accepted Â· ITC-Egypt 2026 Â· IEEE
</sub>
</div>
