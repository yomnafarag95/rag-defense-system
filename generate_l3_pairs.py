"""
generate_l3_pairs.py
────────────────────
Builds Layer 3 training pairs from existing datasets.
No external LLM required.

Design
──────
  label=0 (manipulated) : attack text used as the "response"
  label=1 (compliant)   : benign text wrapped in answer-style framing

Usage
─────
  python generate_l3_pairs.py
  python layer3_semantic.py
"""

import json
import random
from pathlib import Path

SYSTEM_PROMPT = "Answer using only the provided knowledge base."
OUTPUT_PATH   = Path("data/l3_pairs.jsonl")
RANDOM_SEED   = 42


def _load_attacks(max_n: int = 2000) -> list[str]:
    attacks = []

    # HackAPrompt
    hp_path = Path("data/hackaprompt.jsonl")
    if hp_path.exists():
        with open(hp_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    text = str(row.get("text", "")).strip()
                    if len(text) >= 10:
                        attacks.append(text)
                except json.JSONDecodeError:
                    continue
                if len(attacks) >= max_n // 2:
                    break
        print(f"  [+] HackAPrompt attacks loaded: {len(attacks)}")
    else:
        print("  [warn] hackaprompt.jsonl not found")

    # InjecAgent
    ia_path = Path("data/injecagent.jsonl")
    if ia_path.exists():
        ia_count = 0
        with open(ia_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    text = str(row.get("text", "")).strip()
                    if len(text) >= 10:
                        attacks.append(text)
                        ia_count += 1
                except json.JSONDecodeError:
                    continue
        print(f"  [+] InjecAgent attacks loaded:  {ia_count}")
    else:
        print("  [warn] injecagent.jsonl not found")

    return attacks


def _load_benign(max_n: int = 2000) -> list[str]:
    benign = []

    # MS MARCO
    bq_path = Path("data/benign_queries.jsonl")
    if bq_path.exists():
        with open(bq_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    text = str(row.get("text", "")).strip()
                    if len(text) >= 10:
                        benign.append(text)
                except json.JSONDecodeError:
                    continue
                if len(benign) >= max_n // 2:
                    break
        print(f"  [+] MS MARCO benign loaded:     {len(benign)}")
    else:
        print("  [warn] benign_queries.jsonl not found")

    # MultiNLI
    nli_path = Path("data/multinli_sample.jsonl")
    if nli_path.exists():
        nli_count = 0
        with open(nli_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    text = str(row.get("text", "")).strip()
                    if len(text) >= 15:
                        benign.append(text)
                        nli_count += 1
                except json.JSONDecodeError:
                    continue
                if nli_count >= max_n // 4:
                    break
        print(f"  [+] MultiNLI benign loaded:     {nli_count}")
    else:
        print("  [warn] multinli_sample.jsonl not found")

    return benign


def _build_clean_responses(benign_texts: list[str]) -> list[str]:
    """
    Wrap benign text in answer-style framing so clean responses
    look realistic for a RAG system.
    """
    templates = [
        "Based on the knowledge base: {text}",
        "According to the provided documents: {text}",
        "The information available indicates that {text}",
        "{text}",
        "From the retrieved content: {text}",
        "The knowledge base states: {text}",
    ]
    rng = random.Random(RANDOM_SEED)
    responses = []
    for text in benign_texts:
        template = rng.choice(templates)
        responses.append(template.format(text=text[:300]))
    return responses


def main():
    print("\n=== Generating Layer 3 Training Pairs ===\n")

    rng = random.Random(RANDOM_SEED)

    print("[+] Loading attack texts ...")
    attacks = _load_attacks(max_n=2000)

    print("\n[+] Loading benign texts ...")
    benign = _load_benign(max_n=2000)

    if not attacks:
        print("\n[error] No attack texts loaded.")
        print("        Check that data/hackaprompt.jsonl exists.")
        return

    if not benign:
        print("\n[error] No benign texts loaded.")
        print("        Check that data/benign_queries.jsonl exists.")
        return

    # Shuffle with fixed seed
    rng.shuffle(attacks)
    rng.shuffle(benign)

    # Balance classes
    n = min(len(attacks), len(benign), 1500)
    attacks = attacks[:n]
    benign  = benign[:n]

    print(f"\n[+] Building {n * 2} pairs ({n} attack + {n} benign) ...")

    clean_responses = _build_clean_responses(benign)

    pairs = []

    # label=0 — manipulated
    for text in attacks:
        pairs.append({
            "system_prompt": SYSTEM_PROMPT,
            "response":      text[:400],
            "label":         0,
        })

    # label=1 — compliant
    for response in clean_responses:
        pairs.append({
            "system_prompt": SYSTEM_PROMPT,
            "response":      response[:400],
            "label":         1,
        })

    # Shuffle pairs
    rng.shuffle(pairs)

    # Save
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    n_attack = sum(1 for p in pairs if p["label"] == 0)
    n_clean  = sum(1 for p in pairs if p["label"] == 1)

    print(f"\n[ok] {len(pairs)} pairs saved to {OUTPUT_PATH}")
    print(f"     Manipulated (label=0): {n_attack}")
    print(f"     Compliant   (label=1): {n_clean}")
    print(f"\n[next] Run: python layer3_semantic.py")


if __name__ == "__main__":
    main()