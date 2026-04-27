# RAG Defense System

A three-layer prompt injection defense pipeline for RAG systems.

## Setup
pip install -r requirements.txt
python data_loader.py
python layer1_anomaly.py
streamlit run app.py

## Layers
- Layer 1: Anomaly detection (all-MiniLM-L6-v2 + IsolationForest/ECOD/OneClassSVM)
- Layer 2: Intent classification (protectai/deberta-v3-base-prompt-injection-v2)
- Layer 3: Behavioral monitoring (cross-encoder/ms-marco-MiniLM-L-12-v2)

## Evaluation Results
- ADR Standard: 0.655
- ADR Evasion: 0.714
- False Positive Rate: 0.000
- Precision: 1.000
- F1 Standard: 0.792
- Mean Latency: ~360ms
