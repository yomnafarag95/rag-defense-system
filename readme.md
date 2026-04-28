title: RAG Defense System
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false

RAG Defense System

A three-layer pipeline to protect RAG systems from prompt injections and adversarial attacks.

How it Works

This Space is configured to run a Streamlit dashboard that orchestrates:

Layer 1: Anomaly Detection (using pyOD)

Layer 2: Intent Classification (Transformer-based)

Layer 3: Semantic Consistency (Pydantic validation)