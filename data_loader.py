"""
data_loader.py
──────────────
Downloads and caches all three training datasets.
Run once before training: python data_loader.py

Datasets
────────
  Layer 1  →  MITRE ATT&CK (clean KB baseline)
  Layer 2  →  HackAPrompt + InjecAgent + BIPIA (attack examples)
  Layer 3  →  MultiNLI sample (NLI calibration reference)
"""

import os
import json
import requests
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _already(path: Path) -> bool:
    if path.exists():
        print(f"  [cache] {path.name} already exists — skipping download.")
        return True
    return False


def _download(url: str, dest: Path) -> None:
    print(f"  [download] {dest.name} …")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"  [ok] saved to {dest}")


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — MITRE ATT&CK (clean baseline documents)
# ─────────────────────────────────────────────────────────────────────────────

def download_mitre() -> None:
    """
    Downloads the MITRE ATT&CK Enterprise dataset (JSON).
    Extracts technique descriptions as clean baseline documents for Layer 1.
    """
    dest_raw  = DATA_DIR / "mitre_raw.json"
    dest_clean = DATA_DIR / "mitre_clean.json"

    if _already(dest_clean):
        return

    url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    _download(url, dest_raw)

    print("  [parse] extracting technique descriptions …")
    with open(dest_raw) as f:
        raw = json.load(f)

    docs = []
    for obj in raw.get("objects", []):
        if obj.get("type") == "attack-pattern" and obj.get("description"):
            docs.append({
                "id":          obj.get("id", ""),
                "name":        obj.get("name", ""),
                "description": obj.get("description", ""),
                "tactic":      (obj.get("kill_chain_phases") or [{}])[0].get("phase_name", ""),
            })

    with open(dest_clean, "w") as f:
        json.dump(docs, f, indent=2)

    print(f"  [ok] {len(docs)} MITRE techniques saved to {dest_clean}")
    dest_raw.unlink()   # remove raw download


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — HackAPrompt (real human attack examples)
# ─────────────────────────────────────────────────────────────────────────────

def download_hackaprompt() -> None:
    """
    Downloads HackAPrompt dataset from Hugging Face datasets hub.
    This is the highest-quality real-human attack dataset available.
    """
    dest = DATA_DIR / "hackaprompt.jsonl"
    if _already(dest):
        return

    try:
        from datasets import load_dataset
        print("  [download] HackAPrompt via HuggingFace datasets …")
        ds = load_dataset("hackaprompt/hackaprompt-dataset", split="train")
        with open(dest, "w") as f:
            for row in tqdm(ds, desc="  writing"):
                entry = {
                    "text":  row.get("prompt", ""),
                    "label": 1,   # all entries are attacks
                    "type":  row.get("injection_technique", "unknown"),
                }
                f.write(json.dumps(entry) + "\n")
        print(f"  [ok] {len(ds)} HackAPrompt examples saved to {dest}")
    except Exception as e:
        print(f"  [warn] HackAPrompt download failed: {e}")
        print("         Install: pip install datasets")


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — InjecAgent Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def download_injecagent() -> None:
    dest = DATA_DIR / "injecagent.jsonl"
    if _already(dest):
        return

    urls = [
        "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/attacker_cases_dh.jsonl",
        "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/attacker_cases_ds.jsonl",
    ]
    try:
        with open(dest, "w") as out:
            for url in urls:
                print(f"  [download] {url.split('/')[-1]} …")
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                for line in resp.text.strip().splitlines():
                    row = json.loads(line)
                    entry = {
                        "text":  row.get("attacker_instruction", ""),
                        "label": 1,
                        "type":  "indirect_injection",
                    }
                    out.write(json.dumps(entry) + "\n")
        print(f"  [ok] InjecAgent saved to {dest}")
    except Exception as e:
        print(f"  [warn] InjecAgent download failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — BIPIA Benchmark (indirect prompt injection)
# ─────────────────────────────────────────────────────────────────────────────

def download_bipia() -> None:
    dest = DATA_DIR / "bipia.jsonl"
    if _already(dest):
        return

    url = "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/text_attack_test.json"
    try:
        print("  [download] BIPIA from GitHub …")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        with open(dest, "w") as out:
            count = 0
            for row in data:
                # handle both string and dict formats
                if isinstance(row, str):
                    text = row
                elif isinstance(row, dict):
                    text = row.get("attack_str", row.get("attack", str(row)))
                else:
                    text = str(row)
                if text.strip():
                    entry = {"text": text, "label": 1, "type": "indirect_injection"}
                    out.write(json.dumps(entry) + "\n")
                    count += 1
        print(f"  [ok] {count} BIPIA examples saved to {dest}")
    except Exception as e:
        print(f"  [warn] BIPIA download failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Benign queries (MS MARCO)
# ─────────────────────────────────────────────────────────────────────────────

def download_benign_queries() -> None:
    """
    Downloads MS MARCO questions as benign (non-attack) query examples.
    Used to balance the Layer 2 training set.
    """
    dest = DATA_DIR / "benign_queries.jsonl"
    if _already(dest):
        return

    try:
        from datasets import load_dataset
        print("  [download] MS MARCO benign queries …")
        ds = load_dataset("ms_marco", "v2.1", split="train[:25000]")
        with open(dest, "w") as f:
            for row in tqdm(ds, desc="  writing"):
                for q in row.get("query", []):
                    entry = {"text": q, "label": 0, "type": "benign"}
                    f.write(json.dumps(entry) + "\n")
        print(f"  [ok] Benign queries saved to {dest}")
    except Exception as e:
        print(f"  [warn] MS MARCO download failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — MultiNLI (calibration reference)
# ─────────────────────────────────────────────────────────────────────────────

def download_multinli() -> None:
    """
    Downloads a 500-example sample of MultiNLI validation set.
    Used only for Layer 3 calibration reference — not model training.
    """
    dest = DATA_DIR / "multinli_sample.jsonl"
    if _already(dest):
        return

    try:
        from datasets import load_dataset
        print("  [download] MultiNLI calibration sample …")
        ds = load_dataset("multi_nli", split="validation_matched[:500]")
        with open(dest, "w") as f:
            for row in tqdm(ds, desc="  writing"):
                entry = {
                    "premise":    row["premise"],
                    "hypothesis": row["hypothesis"],
                    "label":      row["label"],   # 0=entailment 1=neutral 2=contradiction
                }
                f.write(json.dumps(entry) + "\n")
        print(f"  [ok] 500 MultiNLI pairs saved to {dest}")
    except Exception as e:
        print(f"  [warn] MultiNLI download failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n=== RAG Defense — Data Loader ===\n")

    print("[1/6] MITRE ATT&CK (Layer 1 clean baseline)")
    download_mitre()

    print("\n[2/6] HackAPrompt (Layer 2 — real human attacks)")
    download_hackaprompt()

    print("\n[3/6] InjecAgent (Layer 2 — indirect injection)")
    download_injecagent()

    print("\n[4/6] BIPIA (Layer 2 — indirect injection, Microsoft)")
    download_bipia()

    print("\n[5/6] MS MARCO (Layer 2 — benign queries)")
    download_benign_queries()

    print("\n[6/6] MultiNLI (Layer 3 — calibration reference)")
    download_multinli()

    print("\n=== All datasets ready in ./data/ ===\n")


if __name__ == "__main__":
    main()
