"""
generate_paper_figures.py — Color figures for IEEE paper
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from collections import Counter

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

RESULTS_FILE = "logs/eval_results.jsonl"
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    records = []
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records")
    return records


def make_fig2_roc_pr(records):
    y_true = []
    y_scores = []
    for r in records:
        y_true.append(r.get("true_label", 0))
        y_scores.append(r.get("risk_score", 0.0))

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    if len(set(y_true)) < 2:
        print("WARNING: Only one class. Cannot plot.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # ROC — dark blue
    ax1.plot(fpr, tpr, color='#1565C0', linewidth=1.8,
             label=f'RAG-Shield (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=0.7, alpha=0.4,
             label='Random Baseline')
    ax1.fill_between(fpr, tpr, alpha=0.08, color='#1565C0')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc='lower right')
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.grid(True, alpha=0.15, linewidth=0.5)
    ax1.set_aspect('equal')

    # PR — dark red
    ax2.plot(recall, precision, color='#C62828', linewidth=1.8,
             label=f'RAG-Shield (AUC = {pr_auc:.3f})')
    baseline = y_true.sum() / len(y_true)
    ax2.axhline(y=baseline, color='k', linestyle='--', linewidth=0.7,
                alpha=0.4, label=f'Random ({baseline:.2f})')
    ax2.fill_between(recall, precision, alpha=0.08, color='#C62828')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc='lower left')
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.05])
    ax2.grid(True, alpha=0.15, linewidth=0.5)
    ax2.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_roc_pr.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "fig2_roc_pr.png"))
    print(f"Saved: fig2_roc_pr.pdf / .png  (ROC AUC={roc_auc:.4f}, PR AUC={pr_auc:.4f})")
    plt.close()


def make_fig3_confusion_attribution(records):
    attacks = [r for r in records if r["true_label"] == 1]
    benign = [r for r in records if r["true_label"] == 0]

    std_attacks = attacks[:200] if len(attacks) >= 200 else attacks
    eva_attacks = attacks[200:] if len(attacks) > 200 else []

    std_tp = sum(1 for r in std_attacks if r.get("action", "").lower() in ["blocked", "block"])
    std_fn = len(std_attacks) - std_tp
    std_fp = sum(1 for r in benign if r.get("action", "").lower() in ["blocked", "block"])
    std_tn = len(benign) - std_fp

    eva_tp = sum(1 for r in eva_attacks if r.get("action", "").lower() in ["blocked", "block"])
    eva_fn = len(eva_attacks) - eva_tp

    print(f"  Standard: TP={std_tp}, FN={std_fn}, FP={std_fp}, TN={std_tn}")
    print(f"  Evasion:  TP={eva_tp}, FN={eva_fn}")

    std_blocked = [r for r in std_attacks if r.get("action", "").lower() in ["blocked", "block"]]
    std_layers = Counter(r.get("blocking_layer", "Unknown") for r in std_blocked)
    eva_blocked = [r for r in eva_attacks if r.get("action", "").lower() in ["blocked", "block"]]
    eva_layers = Counter(r.get("blocking_layer", "Unknown") for r in eva_blocked)

    print(f"  Standard layers: {dict(std_layers)}")
    print(f"  Evasion layers:  {dict(eva_layers)}")

    fig = plt.figure(figsize=(6.5, 5.5))

    # Standard Confusion Matrix
    ax1 = fig.add_subplot(2, 2, 1)
    cm_std = np.array([[std_tp, std_fn], [std_fp, std_tn]])
    _plot_confusion_color(ax1, cm_std, f"Standard (n={len(std_attacks)})")

    # Evasion Confusion Matrix
    ax2 = fig.add_subplot(2, 2, 2)
    cm_eva = np.array([[eva_tp, eva_fn], [0, 0]])
    _plot_confusion_color(ax2, cm_eva, f"Evasion (n={len(eva_attacks)})")

    # Standard Layer Attribution
    ax3 = fig.add_subplot(2, 2, 3)
    _plot_attribution_color(ax3, std_layers, std_tp, "Standard Detections")

    # Evasion Layer Attribution
    ax4 = fig.add_subplot(2, 2, 4)
    _plot_attribution_color(ax4, eva_layers, eva_tp, "Evasion Detections")

    plt.tight_layout(h_pad=1.5, w_pad=1.0)
    fig.savefig(os.path.join(OUT_DIR, "fig3_confusion_attribution.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "fig3_confusion_attribution.png"))
    print(f"Saved: fig3_confusion_attribution.pdf / .png")
    plt.close()


def _plot_confusion_color(ax, cm, title):
    # Professional blue colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#FFFFFF', '#E3F2FD', '#90CAF9', '#42A5F5', '#1565C0']
    cmap = LinearSegmentedColormap.from_list('blue_pro', colors)

    ax.imshow(cm, cmap=cmap, aspect='auto', vmin=0, vmax=max(cm.max(), 1))

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = 'white' if val > cm.max() * 0.6 else 'black'
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=14, fontweight='bold', color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Attack", "Pred Benign"], fontsize=8)
    ax.set_yticklabels(["Real Attack", "Real Benign"], fontsize=8)
    ax.set_title(title, fontsize=10, fontweight='bold')


def _plot_attribution_color(ax, layer_counts, total_tp, title):
    if total_tp == 0:
        ax.text(0.5, 0.5, "No detections", ha='center', va='center', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        return

    name_map = {
        "Layer 1 - Anomaly Detection": "L1: Anomaly Det.",
        "Layer 2 - Intent Classifier": "L2: Intent Cls.",
        "Layer 3 - Behavioral Monitor": "L3: Semantic Mon.",
        "Meta Aggregator - Combined Risk": "Meta-Agg",
    }

    # Professional color palette
    color_map = {
        "L1: Anomaly Det.": '#1565C0',   # Dark blue
        "L2: Intent Cls.": '#E65100',     # Dark orange
        "L3: Semantic Mon.": '#2E7D32',   # Dark green
        "Meta-Agg": '#6A1B9A',            # Purple
    }

    labels = []
    sizes = []
    colors = []

    for layer, count in layer_counts.most_common():
        short = name_map.get(layer, layer[:15])
        pct = count / total_tp * 100
        labels.append(f"{short}\n({pct:.0f}%)")
        sizes.append(count)
        colors.append(color_map.get(short, '#757575'))

    wedges, texts = ax.pie(
        sizes, labels=labels, colors=colors,
        startangle=90,
        textprops={'fontsize': 7.5},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
    )
    ax.set_title(title, fontsize=10, fontweight='bold')


if __name__ == "__main__":
    print("=" * 50)
    print("  Generating Color Figures for IEEE Paper")
    print("=" * 50)

    if not os.path.exists(RESULTS_FILE):
        print(f"\nERROR: {RESULTS_FILE} not found!")
        print("Run 'python eval_suite.py --mode all' first.")
        exit(1)

    records = load_data()

    print("\n-- Figure 2: ROC + PR Curves --")
    make_fig2_roc_pr(records)

    print("\n-- Figure 3: Confusion + Attribution --")
    make_fig3_confusion_attribution(records)

    print("\n" + "=" * 50)
    print("  Done! Upload these to Overleaf:")
    print("    figures/fig2_roc_pr.png")
    print("    figures/fig3_confusion_attribution.png")
    print("=" * 50)