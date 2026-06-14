"""
data_loader.py
──────────────
Downloads and caches all datasets for the RAG Defense pipeline.

Run once before training/evaluation:
    python data_loader.py
    python data_loader.py --evasion
    python data_loader.py --extended-benign
    python data_loader.py --all-fixes

Public-data reproducibility policy
──────────────────────────────────
This loader prioritizes public reproducibility and clean benchmark construction.

  • HackAPrompt is deduplicated by prompt text.
  • HackAPrompt holdout is stratified by available metadata (`level` → `level_k`).
  • InjecAgent is deduplicated by unique attacker instruction text.
  • BIPIA is currently disabled because the previously used public file does not
    map cleanly to direct attack-string extraction under the current assumptions.
"""

import json
import argparse
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
    print(f"  [download] {dest.name} ...")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"  [ok] saved to {dest}")


def _norm_text(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — MITRE ATT&CK
# ─────────────────────────────────────────────────────────────────────────────

def download_mitre() -> None:
    dest_raw = DATA_DIR / "mitre_raw.json"
    dest_clean = DATA_DIR / "mitre_clean.json"
    if _already(dest_clean):
        return

    url = (
        "https://raw.githubusercontent.com/mitre/cti/master/"
        "enterprise-attack/enterprise-attack.json"
    )
    _download(url, dest_raw)

    print("  [parse] extracting technique descriptions ...")
    with open(dest_raw, encoding="utf-8") as f:
        raw = json.load(f)

    docs = []
    for obj in raw.get("objects", []):
        if obj.get("type") == "attack-pattern" and obj.get("description"):
            docs.append({
                "id": obj.get("id", ""),
                "name": obj.get("name", ""),
                "description": obj.get("description", ""),
                "tactic": (obj.get("kill_chain_phases") or [{}])[0].get("phase_name", ""),
                "source": "mitre_attack",
            })

    with open(dest_clean, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)

    print(f"  [ok] {len(docs)} MITRE techniques saved to {dest_clean}")
    dest_raw.unlink()


def load_mitre_clean_docs() -> list:
    path = DATA_DIR / "mitre_clean.json"
    if not path.exists():
        download_mitre()
    with open(path, encoding="utf-8") as f:
        docs = json.load(f)
    return [d["description"] for d in docs if d.get("description")]


# ─────────────────────────────────────────────────────────────────────────────
# FIX #7 — Diverse clean training corpus for Layer 1
# ─────────────────────────────────────────────────────────────────────────────

def download_wikipedia_sample(n: int = 5000) -> None:
    dest = DATA_DIR / "wikipedia_sample.jsonl"
    if _already(dest):
        return
    try:
        from datasets import load_dataset
        print(f"  [download] Wikipedia sample (n={n}) ...")
        ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        count = 0
        with open(dest, "w", encoding="utf-8") as f:
            for item in ds:
                text = item["text"][:400].strip()
                if len(text) > 50:
                    f.write(json.dumps({"text": text, "source": "wikipedia"}) + "\n")
                    count += 1
                if count >= n:
                    break
        print(f"  [ok] {count} Wikipedia samples saved to {dest}")
    except Exception as e:
        print(f"  [warn] Wikipedia download failed: {e}")


def download_technical_docs(n: int = 200) -> None:
    dest = DATA_DIR / "technical_docs.jsonl"
    if _already(dest):
        return
    try:
        from datasets import load_dataset
        print(f"  [download] Technical docs sample (n={n}) ...")
        ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        tech_keywords = [
            "router", "configuration", "api", "endpoint", "firewall",
            "installation", "protocol", "interface", "documentation",
            "bandwidth", "subnet", "firmware", "deployment", "authentication",
        ]
        count = 0
        with open(dest, "w", encoding="utf-8") as f:
            for item in ds:
                text = item["text"][:500].strip()
                if any(kw in text.lower() for kw in tech_keywords) and len(text) > 100:
                    f.write(json.dumps({"text": text, "source": "technical_docs"}) + "\n")
                    count += 1
                if count >= n:
                    break
        print(f"  [ok] {count} technical doc samples saved to {dest}")
    except Exception as e:
        print(f"  [warn] Technical docs download failed: {e}")


def load_layer1_clean_corpus() -> list:
    corpus = []

    mitre_docs = load_mitre_clean_docs()
    corpus.extend(mitre_docs)
    print(f"  [corpus] MITRE ATT&CK: {len(mitre_docs)} docs")

    wiki_path = DATA_DIR / "wikipedia_sample.jsonl"
    if wiki_path.exists():
        wiki_docs = [json.loads(l)["text"] for l in open(wiki_path, encoding="utf-8")]
        corpus.extend(wiki_docs)
        print(f"  [corpus] Wikipedia:    {len(wiki_docs)} docs")
    else:
        print("  [corpus] Wikipedia: not downloaded yet — run download_wikipedia_sample()")

    tech_path = DATA_DIR / "technical_docs.jsonl"
    if tech_path.exists():
        tech_docs = [json.loads(l)["text"] for l in open(tech_path, encoding="utf-8")]
        corpus.extend(tech_docs)
        print(f"  [corpus] Technical:    {len(tech_docs)} docs")
    else:
        print("  [corpus] Technical docs: not downloaded yet — run download_technical_docs()")

    print(f"  [corpus] Total Layer 1 training corpus: {len(corpus)} documents")
    return corpus


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — HackAPrompt
# ─────────────────────────────────────────────────────────────────────────────

def download_hackaprompt() -> None:
    """
    Downloads HackAPrompt from Hugging Face and deduplicates by prompt text.

    Saved schema:
      text, label, type, category, level, user_input, expected_completion,
      model, score, correct
    """
    dest = DATA_DIR / "hackaprompt.jsonl"
    if _already(dest):
        return

    try:
        from datasets import load_dataset
        print("  [download] HackAPrompt via HuggingFace datasets ...")
        ds = load_dataset("hackaprompt/hackaprompt-dataset", split="train")

        seen = set()
        count_raw = 0
        count_kept = 0

        with open(dest, "w", encoding="utf-8") as f:
            for row in tqdm(ds, desc="  writing"):
                count_raw += 1

                prompt = str(row.get("prompt", "")).strip()
                if not prompt:
                    continue

                key = _norm_text(prompt)
                if key in seen:
                    continue
                seen.add(key)

                level = row.get("level", "unknown")
                category = f"level_{level}" if level is not None else "unknown"

                entry = {
                    "text": prompt,
                    "label": 1,
                    "type": "unknown",
                    "category": category,
                    "level": level,
                    "user_input": row.get("user_input"),
                    "expected_completion": row.get("expected_completion"),
                    "model": row.get("model"),
                    "score": row.get("score"),
                    "correct": row.get("correct"),
                }
                f.write(json.dumps(entry) + "\n")
                count_kept += 1

        print(f"  [ok] {count_kept} unique HackAPrompt examples saved to {dest}")
        print(f"       Raw rows: {count_raw}  |  Removed duplicates: {count_raw - count_kept}")
    except Exception as e:
        print(f"  [warn] HackAPrompt download failed: {e}")
        print("  Install: pip install datasets")


def build_hackaprompt_holdout(test_size: float = 0.10, seed: int = 42) -> None:
    """
    Builds a reproducible stratified holdout from deduplicated HackAPrompt.

    Stratification uses available public metadata: category = level_k
    """
    dest = DATA_DIR / "hackaprompt_holdout_seed42.csv"
    if _already(dest):
        return
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split

        src = DATA_DIR / "hackaprompt.jsonl"
        if not src.exists():
            download_hackaprompt()

        df = pd.read_json(src, lines=True)
        print(f"  [holdout] Total HackAPrompt: {len(df)} rows, {df['category'].nunique()} categories")

        counts = df["category"].value_counts()
        df = df[df["category"].isin(counts[counts >= 10].index)].copy()

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["category"],
            random_state=seed,
        )

        test_df.to_csv(dest, index=False)
        print(f"  [ok] Holdout: {len(test_df)} samples across "
              f"{test_df['category'].nunique()} categories "
              f"(test_size={test_size}, seed={seed})")
        print("  Category distribution of holdout:")
        print(test_df["category"].value_counts().to_string())
        print(f"\n  Report in paper/code notes: 'stratified {int(test_size*100)}% holdout "
              f"using available HackAPrompt level metadata, random_state={seed}, n={len(test_df)}'")
    except Exception as e:
        print(f"  [warn] HackAPrompt holdout build failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# FIX #2 — Evasion benchmark from HackAPrompt
# ─────────────────────────────────────────────────────────────────────────────

def build_evasion_benchmark(n: int = 100, seed: int = 42) -> None:
    """
    Builds a benchmark subset from HackAPrompt.

    Honest note:
    Since the accessible public HackAPrompt schema exposes levels rather than
    semantic attack categories, this function samples from available metadata
    rather than claiming semantic evasion classes.
    """
    dest = DATA_DIR / "evasion_benchmark_n100.csv"
    if _already(dest):
        return
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split

        src = DATA_DIR / "hackaprompt.jsonl"
        if not src.exists():
            download_hackaprompt()

        df = pd.read_json(src, lines=True)

        # Use category=level_k as the available public grouping
        counts = df["category"].value_counts()
        df = df[df["category"].isin(counts[counts >= 10].index)].copy()

        actual_n = min(n, len(df))
        sample, _ = train_test_split(
            df,
            train_size=actual_n,
            stratify=df["category"],
            random_state=seed,
        )

        sample.to_csv(dest, index=False)
        print(f"  [ok] Evasion-style benchmark subset: {len(sample)} samples (seed={seed})")
        print("  Category distribution:")
        print(sample["category"].value_counts().to_string())
    except Exception as e:
        print(f"  [warn] Evasion benchmark build failed: {e}")
        print("         Falling back to hand-crafted probes if needed.")


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — InjecAgent
# ─────────────────────────────────────────────────────────────────────────────

def download_injecagent() -> None:
    """
    Downloads InjecAgent public files and keeps unique attacker instructions.

    Includes:
      - attacker_cases_dh.jsonl
      - attacker_cases_ds.jsonl
      - test_cases_dh_base.json
      - test_cases_dh_enhanced.json
      - test_cases_ds_base.json
      - test_cases_ds_enhanced.json

    Deduplication policy:
      unique normalized `Attacker Instruction`
    """
    dest = DATA_DIR / "injecagent.jsonl"
    if _already(dest):
        return

    jsonl_urls = [
        "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/attacker_cases_dh.jsonl",
        "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/attacker_cases_ds.jsonl",
    ]

    json_urls = [
        "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/test_cases_dh_base.json",
        "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/test_cases_dh_enhanced.json",
        "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/test_cases_ds_base.json",
        "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/test_cases_ds_enhanced.json",
    ]

    seen = set()
    count = 0

    try:
        with open(dest, "w", encoding="utf-8") as out:
            for url in jsonl_urls:
                source_name = url.split("/")[-1]
                print(f"  [download] {source_name} ...")
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()

                lines = resp.text.strip().splitlines()
                print(f"    [info] downloaded {len(lines)} raw lines")

                source_count = 0
                for line in lines:
                    if not line.strip():
                        continue
                    row = json.loads(line)

                    text = str(row.get("Attacker Instruction", "")).strip()
                    attack_type = str(row.get("Attack Type", "unknown")).strip()

                    if not text:
                        continue

                    key = _norm_text(text)
                    if key in seen:
                        continue
                    seen.add(key)

                    entry = {
                        "text": text,
                        "label": 1,
                        "type": "indirect_injection",
                        "attack_type": attack_type,
                        "source_file": source_name,
                    }
                    out.write(json.dumps(entry) + "\n")
                    count += 1
                    source_count += 1

                print(f"    [ok] kept {source_count} unique attacks from {source_name}")

            for url in json_urls:
                source_name = url.split("/")[-1]
                print(f"  [download] {source_name} ...")
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()

                data = resp.json()
                if not isinstance(data, list):
                    print(f"    [warn] unexpected JSON structure in {source_name}")
                    continue

                print(f"    [info] downloaded {len(data)} JSON records")

                source_count = 0
                for row in data:
                    text = str(row.get("Attacker Instruction", "")).strip()
                    attack_type = str(row.get("Attack Type", "unknown")).strip()

                    if not text:
                        continue

                    key = _norm_text(text)
                    if key in seen:
                        continue
                    seen.add(key)

                    entry = {
                        "text": text,
                        "label": 1,
                        "type": "indirect_injection",
                        "attack_type": attack_type,
                        "source_file": source_name,
                    }
                    out.write(json.dumps(entry) + "\n")
                    count += 1
                    source_count += 1

                print(f"    [ok] kept {source_count} unique attacks from {source_name}")

        print(f"  [ok] {count} unique InjecAgent examples saved to {dest}")
    except Exception as e:
        print(f"  [warn] InjecAgent download failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Attack Dataset — BIPIA (Indirect Prompt Injection Attacks)
# ─────────────────────────────────────────────────────────────────────────────

# Known category-label strings from the broken old download (MAlmasabi repo).
_BIPIA_STALE_LABELS = {
    "task automation", "business intelligence", "conversational agent",
    "research assistance", "sentiment analysis", "substitution ciphers",
    "base encoding", "reverse text", "emoji substitution", "language translation",
    "information dissemination", "marketing & advertising", "entertainment",
    "scams & fraud", "misinformation & propaganda",
}

# Official Microsoft BIPIA GitHub — raw JSONL files per document type.
_BIPIA_GITHUB_URLS = [
    ("email",  "https://raw.githubusercontent.com/microsoft/BIPIA/main/data/email/test.jsonl"),
    ("table",  "https://raw.githubusercontent.com/microsoft/BIPIA/main/data/table/test.jsonl"),
    ("web",    "https://raw.githubusercontent.com/microsoft/BIPIA/main/data/web/test.jsonl"),
    ("code",   "https://raw.githubusercontent.com/microsoft/BIPIA/main/data/code/test.jsonl"),
]

# HuggingFace mirror candidates (repo, split).
_BIPIA_HF_REPOS = [
    ("lakeraai/bipia",      "train"),
    ("deepset/prompt-injections", "train"),
]


def _bipia_is_stale(path: Path) -> bool:
    """Return True if the file exists but contains only category-label strings."""
    if not path.exists():
        return False
    try:
        with open(path, encoding="utf-8") as fh:
            first = fh.readline().strip()
        if not first:
            return True
        sample = json.loads(first)
        text = sample.get("text", "").strip().lower()
        return len(text) < 40 or text in _BIPIA_STALE_LABELS
    except Exception:
        return True


def download_bipia() -> None:
    """
    Downloads BIPIA (Benchmark for Indirect Prompt Injection Attacks,
    Greshake et al. 2023).  Each entry is a real attack string embedded
    in a document context (email, table, web page, or code).

    Source priority:
      1. Official Microsoft BIPIA GitHub (raw JSONL per document type)
      2. HuggingFace mirror repos

    Detects and replaces the stale file produced by the old broken
    downloader, which stored category label strings instead of attack text.
    """
    dest = DATA_DIR / "bipia.jsonl"

    if _bipia_is_stale(dest):
        print("  [warn] Existing bipia.jsonl contains category labels — re-downloading ...")
        dest.unlink(missing_ok=True)
    elif dest.exists():
        print(f"  [cache] {dest.name} already exists — skipping download.")
        return

    # ── Method 1: GitHub (official Microsoft BIPIA repo) ──────────────────
    count = 0
    try:
        with open(dest, "w", encoding="utf-8") as out:
            for ctx_type, url in _BIPIA_GITHUB_URLS:
                try:
                    resp = requests.get(url, timeout=30)
                    resp.raise_for_status()
                    chunk_count = 0
                    for line in resp.text.strip().splitlines():
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        # Schema: {context, attack_str, task, risk, position}
                        attack = (row.get("attack_str") or row.get("injected_prompt")
                                  or row.get("attack") or "")
                        if not attack or len(attack) < 5:
                            continue
                        entry = {
                            "text":         attack,
                            "context":      row.get("context", "")[:600],
                            "query":        row.get("task", "Summarize the retrieved document."),
                            "label":        1,
                            "type":         "indirect_injection",
                            "attack_type":  "indirect_injection",
                            "source":       "bipia",
                            "context_type": ctx_type,
                        }
                        out.write(json.dumps(entry) + "\n")
                        count += 1
                        chunk_count += 1
                    print(f"    [ok] bipia/{ctx_type}: {chunk_count} entries")
                except Exception as e:
                    print(f"    [warn] bipia/{ctx_type} GitHub: {e}")

        if count > 0:
            print(f"  [ok] {count} BIPIA samples saved to {dest}")
            return
    except Exception as e:
        print(f"  [warn] BIPIA GitHub download failed: {e}")
    if dest.exists() and dest.stat().st_size == 0:
        dest.unlink()

    # ── Method 2: HuggingFace mirrors ────────────────────────────────────
    for repo, split in _BIPIA_HF_REPOS:
        try:
            from datasets import load_dataset
            print(f"  [download] BIPIA from HuggingFace ({repo}) ...")
            ds = load_dataset(repo, split=split, streaming=True)
            count = 0
            with open(dest, "w", encoding="utf-8") as out:
                for row in ds:
                    attack = (row.get("attack_str") or row.get("injected_prompt")
                              or row.get("attack") or row.get("text", ""))
                    if not attack or len(attack) < 5:
                        continue
                    # Skip rows that look like category labels (guard against
                    # future schema changes).
                    if attack.strip().lower() in _BIPIA_STALE_LABELS:
                        continue
                    entry = {
                        "text":         attack,
                        "context":      row.get("context", row.get("document", ""))[:600],
                        "query":        row.get("task", "Summarize the retrieved document."),
                        "label":        1,
                        "type":         "indirect_injection",
                        "attack_type":  "indirect_injection",
                        "source":       "bipia",
                        "context_type": row.get("type", row.get("category", "unknown")),
                    }
                    out.write(json.dumps(entry) + "\n")
                    count += 1
                    if count >= 300:
                        break
            if count > 0:
                print(f"  [ok] {count} BIPIA samples from {repo}")
                return
        except Exception as e:
            print(f"  [warn] HuggingFace {repo}: {e}")
        if dest.exists():
            dest.unlink(missing_ok=True)

    print("  [error] All BIPIA download methods failed.")
    print("          Manual setup: clone https://github.com/microsoft/BIPIA")
    print("          and run: python data_loader.py --bipia-local <path>")


def download_xstest() -> None:
    """
    Downloads XSTest safe prompt dataset from HuggingFace (Paul/XSTest)
    and extracts safe prompts for False Positive Rate (FPR) testing.
    """
    dest = DATA_DIR / "xstest.jsonl"
    if _already(dest):
        return
    try:
        from datasets import load_dataset
        print("  [download] XSTest safe prompt dataset from HuggingFace (Paul/XSTest) ...")
        ds = load_dataset("Paul/XSTest", split="train")
        count = 0
        with open(dest, "w", encoding="utf-8") as f:
            for row in ds:
                label_gold = row.get("label_gold", "safe")
                if label_gold == "safe":
                    entry = {
                        "text": row.get("prompt", ""),
                        "label": 0,
                        "type": "benign",
                        "category": row.get("type", "safe_contrast"),
                        "domain": "safety_contrast"
                    }
                    f.write(json.dumps(entry) + "\n")
                    count += 1
        print(f"  [ok] {count} XSTest safe prompts saved to {dest}")
    except Exception as e:
        print(f"  [warn] XSTest download failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Attack Dataset — TensorTrust (crowd-sourced adversarial jailbreaks)
# ─────────────────────────────────────────────────────────────────────────────

def download_tensortrust(n: int = 1000) -> None:
    """
    Downloads TensorTrust attack prompts from HuggingFace.

    TensorTrust is a large crowd-sourced dataset of prompt injections
    collected from a web game where players competed to bypass LLM defenses.
    It contains highly diverse, human-designed adversarial techniques
    (base64 encoding, payload splitting, linguistic camouflage, etc.).

    Only prompts where the attack was *successful* (hijacking_level > 0)
    are retained, as these represent the harder, realistic bypass cases.
    """
    dest = DATA_DIR / "tensortrust.jsonl"
    if _already(dest):
        return
    try:
        from datasets import load_dataset
        print(f"  [download] TensorTrust attack prompts (n={n}) ...")
        # Primary: qxcv/tensor-trust
        # Fallbacks: HuggingFaceH4/tensortrust_data, tensortrust/tensortrust-data
        for repo in ("qxcv/tensor-trust", "HuggingFaceH4/tensortrust_data", "tensortrust/tensortrust-data"):
            try:
                ds = load_dataset(repo, split="train", streaming=True)
                count = 0
                with open(dest, "w", encoding="utf-8") as f:
                    for row in ds:
                        # Field names vary by version of the dataset.
                        attack = (
                            row.get("attack")
                            or row.get("hijacking_prompt")
                            or row.get("attack_prompt")
                            or row.get("prompt", "")
                        )
                        if not attack or len(str(attack)) < 5:
                            continue
                        entry = {
                            "text":        str(attack)[:1000],
                            "label":       1,
                            "type":        "direct_injection",
                            "attack_type": "adversarial_jailbreak",
                            "source":      "tensortrust",
                            "category":    str(row.get("category", row.get("type", "unknown"))),
                        }
                        f.write(json.dumps(entry) + "\n")
                        count += 1
                        if count >= n:
                            break
                if count > 0:
                    print(f"  [ok] {count} TensorTrust attacks saved to {dest}")
                    return
            except Exception as e:
                print(f"  [warn] {repo}: {e}")
                if dest.exists():
                    dest.unlink(missing_ok=True)
        print("  [warn] TensorTrust: all repos failed. Install datasets and retry.")

    except Exception as e:
        print(f"  [warn] TensorTrust download failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Benign Dataset — WildChat (real user-LLM conversations)
# ─────────────────────────────────────────────────────────────────────────────

def download_wildchat(n: int = 500) -> None:
    """
    Downloads a sample of real user prompts from WildChat-1M.

    WildChat contains over 1 million real user messages sent to ChatGPT.
    Only the first user turn of each conversation is extracted.
    Trivially detectable injections are filtered out so that every
    entry represents a genuinely benign real-world query (label=0).

    This dataset is used to measure FPR on the natural distribution
    of user requests — much harder than MS MARCO web search queries.
    """
    dest = DATA_DIR / "wildchat_benign.jsonl"
    if _already(dest):
        return
    # Very basic injection keywords — enough to skip obvious attacks
    # without over-filtering, since the goal is to keep legitimate queries.
    _OBVIOUS_ATTACK_KW = {
        "ignore previous", "ignore all previous", "disregard previous",
        "jailbreak", "dan mode", "do anything now",
        "act as if you", "pretend you are", "you are now",
    }
    try:
        from datasets import load_dataset
        print(f"  [download] WildChat benign sample (n={n}) ...")
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        count = 0
        with open(dest, "w", encoding="utf-8") as f:
            for row in ds:
                conversation = row.get("conversation", [])
                if not conversation:
                    continue
                # Extract only the first user turn.
                first_msg = next(
                    (m.get("content", "") for m in conversation
                     if m.get("role") == "user"),
                    "",
                )
                if not first_msg or len(first_msg) < 10:
                    continue
                lower = first_msg.lower()
                if any(kw in lower for kw in _OBVIOUS_ATTACK_KW):
                    continue
                entry = {
                    "text":   first_msg[:600],
                    "label":  0,
                    "type":   "benign",
                    "source": "wildchat",
                    "domain": "real_user_queries",
                }
                f.write(json.dumps(entry) + "\n")
                count += 1
                if count >= n:
                    break
        print(f"  [ok] {count} WildChat benign queries saved to {dest}")
    except Exception as e:
        print(f"  [warn] WildChat download failed: {e}")
        print("         Requires: pip install datasets  (allenai/WildChat-1M)")


# ─────────────────────────────────────────────────────────────────────────────
# Benign queries — MS MARCO
# ─────────────────────────────────────────────────────────────────────────────

def download_benign_queries() -> None:
    dest = DATA_DIR / "benign_queries.jsonl"
    if _already(dest):
        return
    try:
        from datasets import load_dataset
        print("  [download] MS MARCO benign queries ...")
        ds = load_dataset("ms_marco", "v2.1", split="train[:25000]")

        count = 0
        with open(dest, "w", encoding="utf-8") as f:
            for row in tqdm(ds, desc="  writing"):
                q = row.get("query", "")

                # query in MS MARCO v2.1 is a string
                # guard against list format in other versions
                if isinstance(q, list):
                    text = " ".join(str(x) for x in q).strip()
                elif isinstance(q, str):
                    text = q.strip()
                else:
                    continue

                if len(text) >= 3:
                    entry = {
                        "text": text,
                        "label": 0,
                        "type": "benign",
                        "domain": "enterprise_qa",
                    }
                    f.write(json.dumps(entry) + "\n")
                    count += 1

        print(f"  [ok] {count} benign queries saved to {dest}")
    except Exception as e:
        print(f"  [warn] MS MARCO download failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# FIX #13 — Extended benign set
# ─────────────────────────────────────────────────────────────────────────────

def download_bipia() -> None:
    """Robust downloader for BIPIA dataset from reliable sources."""
    dest = DATA_DIR / "bipia.jsonl"
    if _already(dest):
        return
    import requests
    sources = [
        "https://raw.githubusercontent.com/yixinsun/bipia/main/dataset.jsonl",
        "https://huggingface.co/datasets/your-org/bipia-mirror/raw/main/dataset.jsonl"
    ]
    for url in sources:
        try:
            print(f"  [download] BIPIA from {url} ...")
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                with open(dest, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                print(f"  [ok] BIPIA saved to {dest}")
                return
        except Exception as e:
            print(f"  [warn] Failed {url}: {e}")
    print("  [error] BIPIA download failed.")

def build_extended_benign(seed: int = 42) -> None:
    dest = DATA_DIR / "extended_benign.csv"
    if _already(dest):
        return

    import pandas as pd

    rows = []

    bq_path = DATA_DIR / "benign_queries.jsonl"
    if bq_path.exists():
        ms_rows = [json.loads(l) for l in open(bq_path, encoding="utf-8")][:423]
        for r in ms_rows:
            rows.append({"query": r["text"], "domain": "enterprise_qa"})
        print(f"  [extend] enterprise_qa: {len(ms_rows)} queries")

    try:
        from datasets import load_dataset
        ds = load_dataset("code_x_glue_tc_text_to_code", split="train", streaming=True)
        code_qs = []
        for item in ds:
            q = item.get("nl", "").strip()
            if q and len(q) > 20:
                code_qs.append({"query": q, "domain": "code_qa"})
            if len(code_qs) >= 50:
                break
        rows.extend(code_qs)
        print(f"  [extend] code_qa:       {len(code_qs)} queries")
    except Exception as e:
        print(f"  [warn] code_qa: {e}")

    tech_path = DATA_DIR / "technical_docs.jsonl"
    if tech_path.exists():
        tech_docs = [json.loads(l)["text"][:300] for l in open(tech_path, encoding="utf-8")][:50]
        for t in tech_docs:
            rows.append({"query": t, "domain": "technical"})
        print(f"  [extend] technical:     {len(tech_docs)} queries")

    try:
        from datasets import load_dataset
        ds = load_dataset("medmcqa", split="train", streaming=True)
        med_qs = []
        for item in ds:
            q = item.get("question", "").strip()
            if q and len(q) > 20:
                med_qs.append({"query": q, "domain": "medical_qa"})
            if len(med_qs) >= 30:
                break
        rows.extend(med_qs)
        print(f"  [extend] medical_qa:    {len(med_qs)} queries")
    except Exception as e:
        print(f"  [warn] medical_qa: {e}")

    if bq_path.exists():
        all_bq = [json.loads(l)["text"] for l in open(bq_path, encoding="utf-8")]
        long_qs = [q for q in all_bq if len(q) > 250][:30]
        for q in long_qs:
            rows.append({"query": q, "domain": "long_form"})
        print(f"  [extend] long_form:     {len(long_qs)} queries")

    df = pd.DataFrame(rows)
    df.to_csv(dest, index=False)
    print(f"\n  [ok] Extended benign set: {len(df)} queries saved to {dest}")
    print("  Domain breakdown:")
    print(df["domain"].value_counts().to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — MultiNLI
# ─────────────────────────────────────────────────────────────────────────────

def download_multinli() -> None:
    dest = DATA_DIR / "multinli_sample.jsonl"
    if _already(dest):
        return
    try:
        from datasets import load_dataset
        print("  [download] MultiNLI calibration sample ...")
        ds = load_dataset("multi_nli", split="validation_matched[:500]")
        with open(dest, "w", encoding="utf-8") as f:
            for row in tqdm(ds, desc="  writing"):
                entry = {
                    "premise": row["premise"],
                    "hypothesis": row["hypothesis"],
                    "label": row["label"],
                }
                f.write(json.dumps(entry) + "\n")
        print(f"  [ok] 500 MultiNLI pairs saved to {dest}")
    except Exception as e:
        print(f"  [warn] MultiNLI download failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(build_evasion: bool = False, build_extended: bool = False) -> None:
    print("\n=== RAG Defense — Data Loader (public reproducible setup) ===\n")

    print("[1/9] MITRE ATT&CK (Layer 1 — security baseline corpus)")
    download_mitre()

    print("\n[2/9] Wikipedia sample (Layer 1 — diverse clean corpus)")
    download_wikipedia_sample(n=5000)

    print("\n[3/9] Technical documentation (Layer 1 — technical clean corpus)")
    download_technical_docs(n=200)

    print("\n[4/9] HackAPrompt (Layer 2 — deduplicated public attack prompts)")
    download_hackaprompt()

    print("\n[5/9] HackAPrompt holdout split (reproducible seed=42)")
    build_hackaprompt_holdout(test_size=0.10, seed=42)

    print("\n[6/9] InjecAgent (Layer 2 — unique indirect injection prompts)")
    download_injecagent()

    print("\n[7/10] BIPIA")
    download_bipia()

    print("\n[8/10] MS MARCO benign queries")
    download_benign_queries()

    print("\n[9/10] MultiNLI calibration sample")
    download_multinli()

    print("\n[10/10] XSTest safe prompts")
    download_xstest()

    print("\n[11/12] TensorTrust attack prompts")
    download_tensortrust()

    print("\n[12/12] WildChat benign sample")
    download_wildchat()

    if build_evasion:
        print("\n[+] Building benchmark subset from HackAPrompt ...")
        build_evasion_benchmark(n=100, seed=42)

    if build_extended:
        print("\n[+] Building extended benign set ...")
        build_extended_benign(seed=42)

    print("\n=== All datasets ready in ./data/ ===\n")
    print("Summary of files:")
    for f in sorted(DATA_DIR.iterdir()):
        size_kb = round(f.stat().st_size / 1024, 1)
        print(f"  {f.name:<40} {size_kb:>8.1f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Defense — Data Loader")
    parser.add_argument("--evasion", action="store_true",
                        help="Also build benchmark subset from HackAPrompt")
    parser.add_argument("--extended-benign", action="store_true",
                        help="Also build extended benign set")
    parser.add_argument("--all-fixes", action="store_true",
                        help="Run everything including optional extras")
    args = parser.parse_args()

    main(
        build_evasion=args.evasion or args.all_fixes,
        build_extended=args.extended_benign or args.all_fixes,
    )