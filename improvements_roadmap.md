# RAG-Shield Improvement & Research Roadmap

This document outlines the proposed roadmap to optimize **RAG-Shield**'s security detection rate (ADR), reduce latency, and strengthen its comparison against competitive baselines (such as LLM-Guard and PromptGuard) for publication and enterprise-grade deployment.

---

## 📊 Performance Comparison Matrix (Estimates)

| Metric | Current RAG-Shield | RAG-Shield + Local Improvements | RAG-Shield + Advanced Research | Competitor (LLM-Guard) | Competitor (PromptGuard) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Direct Attack Detection (ADR)** | ~68.7% | **~72% to 75%** | **~85% to 90%** | ~70% | ~63% |
| **Indirect / Poisoning ADR** | ~0.0% | ~0.0% | **~92%** *(via Honeypots)* | ~15% | ~10% |
| **Obfuscated Attack ADR** | ~50.0% | ~50.0% | **~95%** *(via Decoders)* | ~20% | ~5% |
| **Benign False Positive Rate (FPR)**| 2.5% | **< 2.0%** | **< 1.0%** *(via tuning)* | ~5.0% | ~8.0% |
| **Average Latency** | ~470ms | **~130ms** *(INT8 ONNX)*| **~90ms** *(small model)* | ~250ms | ~150ms |

---

## 🟢 Tier 1: Quick Wins (Low Effort, High Impact)

### 1. Differentiated Keyword Boosts
* **Concept:** Currently, any keyword match in Layer 1 (Keyword/Anomaly Detection) applies a uniform boost of `0.40`. However, specific DAN/jailbreak phrases (e.g., `"ignore all previous instructions"`, `"You are now DAN"`) are high-confidence indicators of malicious intent.
* **Action:** 
  * Classify keywords into "High-Confidence Jailbreaks" and "Medium-Confidence Indicators".
  * Assign a boost score of `0.55` to high-confidence terms (automatically crossing the meta-block threshold of `0.35`) and `0.30` to medium-confidence terms.
* **File to modify:** [keyword_detector.py](file:///c:/Users/DPQUCS250122/Downloads/rag-defense-system/keyword_detector.py)

---

## 🟡 Tier 2: Latency & Architectural Optimization (Medium Effort)

### 2. Parallel Layer Execution
* **Concept:** RAG-Shield evaluates inputs sequentially (Layer 1 ➔ Layer 2 ➔ Layer 3). If Layer 1 and Layer 2 run concurrently, the pipeline latency will be capped by the slower component (Layer 2 DeBERTa, ~350-400ms) rather than their sum.
* **Action:**
  * Use `asyncio` or a thread pool (`ThreadPoolExecutor`) in `orchestrator.py` to execute Layer 1 (Anomaly Detection + Keywords) and Layer 2 (DeBERTa Intent Classification) in parallel.
* **Impact:** Cuts baseline latency by **20–30%** (saving ~50–100ms per request).

### 3. INT8 Quantization for DeBERTa
* **Concept:** The off-the-shelf DeBERTa-v3 model runs in FP32/FP16, causing high CPU/GPU inference latency. Quantizing the model weights to 8-bit integer precision (INT8) yields significant performance gains.
* **Action:**
  * Convert the DeBERTa HuggingFace checkpoint to ONNX runtime format with INT8 quantization using `optimum`.
  * Update `layer2_classifier.py` to use ONNX Runtime CPU execution provider.
* **Impact:** Drops DeBERTa inference latency from **~400ms to ~130ms** on standard CPU instances.

---

## 🔴 Tier 3: Model and Benchmark Expansion (High Effort)

### 4. Custom Fine-Tuned Domain Classifier
* **Concept:** Instead of relying on a general-purpose off-the-shelf classifier, fine-tune a smaller, highly optimized model (e.g., `distilroberta-base` or `deberta-v3-small`) specifically on the prompt injection corpus.
* **Action:**
  * Run a training pipeline to fine-tune the classifier on standard prompt injection datasets.
  * Save the custom model checkpoint and load it locally.
* **Impact:** 
  * Boosts baseline ADR from **~68% to ~82%+**.
  * Reduces model parameter footprint, cutting inference latency down to **~80-100ms**.

### 5. Benchmark Comparison with Published Baselines
* **Concept:** Reviewers want to see how RAG-Shield performs directly against established open-source guardrails like **LLM-Guard** or **PromptGuard**.
* **Action:**
  * Install `llm-guard` and run their prompt injection detectors on the same evaluation dataset splits (131 attacks, 553 benigns).
  * Report a head-to-head comparison matrix.

---

## 🏆 Tier 4: Advanced Research Improvements (State-of-the-Art)

### 6. Active Defense via "Canary/Honeypot Documents"
* **Concept:** Adversaries can dynamically probe the system to bypass passive classifiers. This improvement introduces an active honey-pot defense in the vector database.
* **Action:**
  * Inject decoy "honeypot" document chunks into the database containing dummy information and unique, randomized canary tokens (e.g., `[CANARY_REF_8492]`).
  * If a user's query or the generated output contains or references this specific canary token without it being part of the legitimate context, trigger an immediate, high-confidence block.
* **Impact:** Solves the **Indirect Prompt Injection** problem (attacks hidden in retrieved documents), which is the single hardest vulnerability to defend in RAG pipelines today.

### 7. Stateful & Multi-Turn Attack Tracking
* **Concept:** Attackers use multi-turn dialogue to gradually guide a model into jailbreak territory (e.g., *Crescendo* attacks), where no single prompt looks malicious in isolation.
* **Action:**
  * Maintain a sliding window of user-query features (Layer 1, 2, and 3 outputs) and model them as a time-series anomaly detection task.
  * If the cumulative "drift score" across 3 turns exceeds a threshold, raise an alert/block.
* **Impact:** Protects against complex, stateful conversational attacks.

### 8. Robustness to Obfuscation (Automatic Decoder Pre-processors)
* **Concept:** Attackers bypass regex/keywords by using Base64, ROT13, Morse code, or Leetspeak (e.g., `1gn0r3 pr3v10us instuct10ns`).
* **Action:**
  * Introduce a preprocessing layer that checks for high-entropy strings, automatically decodes common formats (Base64, Hex, Binary), and normalizes Leetspeak characters back to standard English before feeding the string to DeBERTa.
* **Impact:** Boosts ADR on evasion/obfuscated datasets from **~50% to ~95%+**.
