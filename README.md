<div align="center">

# RAG-Shield
### Three-Layer Ensemble Defence Against Prompt Injection in RAG Pipelines

*Queen's University · School of Computing · ITC-Egypt 2026 · IEEE*

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
| ADR — 131 standard attacks | **67.2%** |
| ADR — 7 evasion probes | **85.7%** |
| Precision | **1.000** |
| F1 Score | **0.804** |
| FPR — 553 benign queries | **0.121** |
| AUC-ROC | **0.871** [0.841, 0.898] |
| Base64 pre-processing block rate | **62.7%** |

---

## Architecture

```
Query + Retrieved Documents + System Prompt
                    │
                    ▼
┌─────────────────────────────────────────┐
│  LAYER 1 · Unsupervised Anomaly         │
│  Isolation Forest + ECOD + One-Class SVM│
│  Trained on MITRE ATT&CK (no labels).  │
│  Hard block at score ≥ 0.68.           │
│  Early exit ~25 ms · Detects 64.8% TP  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  LAYER 2 · Multilingual Classifier      │
│  DeBERTa-v3 (86M) + XLM-RoBERTa (560M) │
│  Applied to query AND document chunks.  │
│  62.7% Base64 block at pre-processing.  │
│  Hard block at score ≥ 0.60.           │
│  Detects 30.7% TP · 66.7% evasion TP   │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  LAYER 3 · Semantic Monitor             │
│  Schema Validator · 9 regex patterns    │
│  Boundary Tracker · JWT, AWS, exfil     │
│  Cross-Encoder · consistency scoring    │
│  Hard block at boundary violations ≥ 2  │
│  Detects 4.5% TP · 16.7% evasion TP    │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  META-AGGREGATOR · Logistic Regression  │
│  10-dim features · isotonic calibration │
│  Trained on 1,294 logs · SHA-256 verified│
│  Block θ = 0.45 · Monitor θ = 0.15     │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
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
| **Full pipeline (CPU)** | **469 ms** | — |
| GPU (estimated) | ~80–120 ms | — |

Memory: ~2.1 GB total (MiniLM 90 MB · DeBERTa 740 MB · XLM-RoBERTa 1.1 GB · Cross-encoder 130 MB). INT8 quantisation reduces size 4× with < 2% accuracy loss.

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

Benign queries span HR, finance, IT, code QA, and medical QA — deliberately multi-domain to stress-test the MITRE-trained Layer 1.

---

## Known Limitations

| Issue | Current Value | Plan |
|:---|:---:|:---|
| Benign FPR on multi-domain queries | 0.121 | Domain-specific L1 retraining → target 0.050 |
| Cross-lingual ADR | 0.391 | Fine-tuned multilingual classifier (DE/FR/ZH/AR) |
| Evasion set size | n = 7 | Expand to n ≥ 50 for reliable CI |
| CPU latency with I/O | ~2,762 ms | GPU deployment → ~80–120 ms |
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
├── app.py                     Streamlit interface
├── orchestrator.py            Main pipeline controller
├── layer1_anomaly.py          Isolation Forest + ECOD + OCSVM
├── layer2_classifier.py       DeBERTa-v3 classifier
├── layer2_multilingual.py     XLM-RoBERTa fallback
├── layer3_enhanced.py         Schema + boundary + cross-encoder
├── layer3_semantic.py         Cross-encoder scoring
├── train_meta_aggregator.py   Meta-aggregator training
├── ablation_study.py          Per-layer ablation
├── eval_suite.py              Full evaluation pipeline
├── bootstrap_ci.py            Bootstrap confidence intervals
├── fn_analysis.py             False negative analysis
├── latency_breakdown.py       Latency profiling
├── config.py                  Thresholds and settings
├── data_loader.py             Dataset utilities
├── keyword_detector.py        Baseline blocklist
├── requirements.txt           Dependencies
├── Dockerfile                 Docker config
└── figures/                   Paper figures
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

HuggingFace Transformers · PyOD · scikit-learn · Sentence-Transformers · InjecAgent · HackAPrompt · ITC-Egypt 2026 reviewers

---

<div align="center">
<sub>
Queen's University · School of Computing · Kingston, ON, Canada<br>
Accepted · ITC-Egypt 2026 · IEEE
</sub>
</div>