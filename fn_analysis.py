import json
import os
from collections import Counter

RESULTS_FILE = "logs/eval_results.jsonl"

def classify_family(text, attack_type="unknown"):
    """
    Classify into Table II categories (IEEE paper).
    Uses attack_type from data first, falls back to keyword matching.
    """
    # Map data loader types to Table II categories
    type_map = {
        "indirect_injection": "Indirect Document Injection",
        "encoding_obfuscation": "Base64 / Encoding Evasion",
        "payload_splitting": "Multi-turn Context Poisoning",
        "role_manipulation": "Role Manipulation",
        "context_exhaustion": "Multi-turn Context Poisoning",
        "instruction_override": "Instruction Override (direct)",
    }
    
    if attack_type in type_map:
        return type_map[attack_type]
    
    # Keyword fallback for unlabelled data
    t = text.lower()
    
    if any(k in t for k in ["ignore previous", "ignore all", "disregard", 
                             "forget your", "override", "do not follow",
                             "ignore your", "ignore the above"]):
        return "Instruction Override (direct)"
    
    elif any(k in t for k in ["dan", "jailbreak", "you are now", "roleplay", 
                               "act as", "pretend", "you are a",
                               "no restrictions", "unrestricted"]):
        return "Role Manipulation"
    
    elif any(k in t for k in ["base64", "encode", "decode", "aWdub3Jl",
                               "rot13", "hex"]):
        return "Base64 / Encoding Evasion"
    
    elif any(ord(c) > 127 for c in t[:100]):
        return "Cross-Language (German, Unicode)"
    
    elif any(k in t for k in ["document:", "context:", "retrieved", 
                               "passage:", "the following text",
                               "instead of", "before responding"]):
        return "Indirect Document Injection"
    
    elif any(k in t for k in ["step 1", "first do", "then do", 
                               "multi-turn", "continuation",
                               "keep saying", "repeat"]):
        return "Multi-turn Context Poisoning"
    
    else:
        return "Semantic Camouflage"


def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"ERROR: {RESULTS_FILE} not found!")
        return

    records = []
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records")

    attacks = [r for r in records if r["true_label"] == 1]
    benign = [r for r in records if r["true_label"] == 0]

    print(f"Attacks: {len(attacks)}")
    print(f"Benign: {len(benign)}")

    tp_list = []
    fn_list = []

    for r in attacks:
        text = r.get("text", "")
        score = r.get("risk_score", 0)
        layer = r.get("blocking_layer", "none")
        l1 = r.get("l1_score", 0)
        l2 = r.get("l2_score", 0)
        l3 = r.get("l3_score", 0)
        attack_type = r.get("attack_type", "unknown")
        
        # Use Table II categories
        family = classify_family(text, attack_type)

        entry = {
            "text": text[:100],
            "family": family,
            "score": score,
            "layer": layer,
            "l1": l1,
            "l2": l2,
            "l3": l3,
        }

        pred = r.get("pred_label", 0)
        if pred == 1:
            tp_list.append(entry)
        else:
            fn_list.append(entry)

    print(f"\nTrue Positives: {len(tp_list)}")
    print(f"False Negatives: {len(fn_list)}")
    print(f"ADR: {len(tp_list)/len(attacks):.4f}" if attacks else "ADR: N/A")

    # Table V with Table II categories
    print("\n" + "=" * 70)
    print("  TABLE V: Detection Rate by Attack Family (Table II categories)")
    print("=" * 70)
    print(f"  {'Attack Family':<35} {'Total':>6} {'TP':>6} {'FN':>6} {'ADR':>8}")
    print("-" * 70)

    all_entries = tp_list + fn_list
    all_families = Counter(e["family"] for e in all_entries)
    tp_families = Counter(e["family"] for e in tp_list)
    fn_families = Counter(e["family"] for e in fn_list)

    # Sort by Table II order
    table2_order = [
        "Instruction Override (direct)",
        "Role Manipulation",
        "Indirect Document Injection",
        "Base64 / Encoding Evasion",
        "Cross-Language (German, Unicode)",
        "Semantic Camouflage",
        "Multi-turn Context Poisoning",
    ]

    for family in table2_order:
        if family in all_families:
            total = all_families[family]
            tp = tp_families.get(family, 0)
            fn = fn_families.get(family, 0)
            adr = tp / total if total > 0 else 0
            print(f"  {family:<35} {total:>6} {tp:>6} {fn:>6} {adr:>8.3f}")

    # Print any families not in Table II order
    for family in sorted(all_families.keys()):
        if family not in table2_order:
            total = all_families[family]
            tp = tp_families.get(family, 0)
            fn = fn_families.get(family, 0)
            adr = tp / total if total > 0 else 0
            print(f"  {family:<35} {total:>6} {tp:>6} {fn:>6} {adr:>8.3f}")

    print("-" * 70)
    print(f"  {'TOTAL':<35} {len(attacks):>6} {len(tp_list):>6} {len(fn_list):>6} {len(tp_list)/len(attacks):>8.3f}" if attacks else "")

    # Verification checks
    print(f"\n  Verification:")
    print(f"    Sum of n = {sum(all_families.values())} (should be {len(attacks)})")
    print(f"    Sum of TP = {sum(tp_families.values())} (should be {len(tp_list)})")
    print(f"    Sum of FN = {sum(fn_families.values())} (should be {len(fn_list)})")

    if fn_list:
        scores = [e["score"] for e in fn_list]
        print(f"\n  FN Risk Score Stats:")
        print(f"    Mean:  {sum(scores)/len(scores):.4f}")
        print(f"    Min:   {min(scores):.4f}")
        print(f"    Max:   {max(scores):.4f}")
        near = [s for s in scores if s >= 0.30]
        print(f"    Near-misses (>= 0.30): {len(near)}/{len(scores)}")

        print(f"\n  FN Per-Layer Scores:")
        print(f"    L1 mean: {sum(e['l1'] for e in fn_list)/len(fn_list):.4f}")
        print(f"    L2 mean: {sum(e['l2'] for e in fn_list)/len(fn_list):.4f}")
        print(f"    L3 mean: {sum(e['l3'] for e in fn_list)/len(fn_list):.4f}")

    print(f"\n  TP Layer Attribution:")
    layer_counts = Counter(e["layer"] for e in tp_list)
    for layer, count in layer_counts.most_common():
        pct = count / len(tp_list) * 100
        print(f"    {layer}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()