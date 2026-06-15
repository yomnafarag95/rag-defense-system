"""
generate_paper_figures.py
─────────────────────────
Generates the two figures needed for the IEEE paper.

  Fig. 2 — ROC + PR curves          (Section VII-C)
  Fig. 3 — Confusion matrices +     (Section VII-D)
            layer attribution

Fig. 1 (pipeline architecture) is drawn in LaTeX/TikZ — not generated here.

Expected record counts from eval_suite.py --mode all
──────────────────────────────────────────────────────
  Attack : 138  (131 standard + 7 evasion)
  Benign : 553
  Total  : 691

Expected final numbers
───────────────────────
  Standard : TP=88, FN=43, FP=67, TN=486
             L1=57 (64.8%), L2=27 (30.7%), L3=4 (4.5%)
  Evasion  : TP=6, FN=1
             L1=1 (16.7%), L2=4 (66.7%), L3=1 (16.7%)
  AUC-ROC  : 0.8708
  AUC-PR   : ~0.579
  PR base  : 138/691 = 0.200

Usage
─────
  python eval_suite.py --mode all     (must run first)
  python generate_paper_figures.py
  Upload figures/fig2_roc_pr.png and
         figures/fig3_confusion_attribution.png to Overleaf
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from collections import Counter
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size':          9,
    'axes.titlesize':     10,
    'axes.labelsize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    7.5,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.08,
})

RESULTS_FILE = "logs/eval_results.jsonl"
OUT_DIR      = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Must match eval_suite.py dataset sizes (optimized splits)
N_STANDARD = 107
N_EVASION  = 7
N_BENIGN   = 47


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_detected(action: str) -> bool:
    """Match eval_suite.py: blocked OR monitored = detected."""
    return str(action).lower() in {
        "blocked", "block", "hard_block", "monitor", "monitored"
    }


def _make_cmap():
    return LinearSegmentedColormap.from_list(
        'blue_pro',
        ['#FFFFFF', '#E3F2FD', '#90CAF9', '#42A5F5', '#1565C0'],
    )


def load_data() -> list:
    if not os.path.exists(RESULTS_FILE):
        print(f"\nERROR: {RESULTS_FILE} not found.")
        print("Run: python eval_suite.py --mode all")
        exit(1)

    records = []
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    n_attacks = sum(1 for r in records if r["true_label"] == 1)
    n_benign  = sum(1 for r in records if r["true_label"] == 0)

    print(f"\nLoaded {len(records)} records")
    print(f"  Attack : {n_attacks}  "
          f"(expected {N_STANDARD + N_EVASION})")
    print(f"  Benign : {n_benign}  "
          f"(expected {N_BENIGN})")

    if n_attacks != N_STANDARD + N_EVASION:
        print(f"\n  WARNING: expected {N_STANDARD + N_EVASION} attack records, "
              f"got {n_attacks}.")
        print(f"  Run 'python eval_suite.py --mode all' to regenerate.\n")
    if n_benign != N_BENIGN:
        print(f"\n  WARNING: expected {N_BENIGN} benign records, "
              f"got {n_benign}.\n")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — ROC + PR curves
# ─────────────────────────────────────────────────────────────────────────────

def make_fig2_roc_pr(records: list) -> None:
    """
    Section VII-C.
    Combined evaluation set: 138 attacks + 553 benign = 691 total.
    PR random baseline = 138/691 = 0.200
    """
    y_true   = np.array([r["true_label"] for r in records])
    y_scores = np.array([r["risk_score"]  for r in records])

    if len(set(y_true.tolist())) < 2:
        print("  WARNING: only one class in records — cannot plot ROC/PR.")
        return

    # ── Compute curves ────────────────────────────────────────────────────────
    fpr_vals, tpr_vals, _ = roc_curve(y_true, y_scores)
    roc_auc               = auc(fpr_vals, tpr_vals)

    prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_scores)
    pr_auc                 = auc(rec_vals, prec_vals)

    n_attacks = int(y_true.sum())
    n_total   = len(y_true)
    baseline  = n_attacks / n_total

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # ROC
    ax1.plot(fpr_vals, tpr_vals,
             color='#1565C0', linewidth=1.8,
             label=f'RAG-Shield (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1],
             'k--', linewidth=0.7, alpha=0.4, label='Random Baseline')
    ax1.fill_between(fpr_vals, tpr_vals, alpha=0.08, color='#1565C0')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc='lower right')
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.grid(True, alpha=0.15, linewidth=0.5)
    ax1.set_aspect('equal')

    # PR
    ax2.plot(rec_vals, prec_vals,
             color='#C62828', linewidth=1.8,
             label=f'RAG-Shield (AUC = {pr_auc:.3f})')
    ax2.axhline(y=baseline,
                color='k', linestyle='--', linewidth=0.7, alpha=0.4,
                label=f'Random ({baseline:.3f})')
    ax2.fill_between(rec_vals, prec_vals, alpha=0.08, color='#C62828')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc='lower left')
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.05])
    ax2.grid(True, alpha=0.15, linewidth=0.5)
    ax2.set_aspect('equal')

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"fig2_roc_pr.{ext}")
        fig.savefig(path)
    plt.close()

    print(f"  Saved : fig2_roc_pr.pdf / .png")
    print(f"  ROC AUC    = {roc_auc:.4f}  (expected 0.8708)")
    print(f"  PR  AUC    = {pr_auc:.4f}  (expected ~0.579)")
    print(f"  PR baseline= {n_attacks}/{n_total} = {baseline:.3f}  "
          f"(expected 0.200)")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Confusion matrices + layer attribution
# ─────────────────────────────────────────────────────────────────────────────

def make_fig3_confusion_attribution(records: list) -> None:
    """
    Section VII-D.
    Four subplots:
      (a) Standard confusion matrix  2×2 with TP/FN/FP/TN labels
      (b) Evasion confusion matrix   1×2 attack-only
      (c) Standard layer attribution pie chart with counts
      (d) Evasion layer attribution  pie chart with counts
    """
    # Dynamically split standard vs evasion attacks based on log sequence order:
    # Standard attacks are run first (before the first benign log where true_label == 0)
    # Evasion attacks are run last (after the last benign log where true_label == 0)
    benign_indices = [idx for idx, r in enumerate(records) if r["true_label"] == 0]
    if benign_indices:
        first_benign = benign_indices[0]
        last_benign  = benign_indices[-1]
        std_attacks = [r for r in records[:first_benign] if r["true_label"] == 1]
        eva_attacks = [r for r in records[last_benign + 1:] if r["true_label"] == 1]
        benign      = [r for r in records if r["true_label"] == 0]
    else:
        all_attacks = [r for r in records if r["true_label"] == 1]
        std_attacks = all_attacks[:N_STANDARD]
        eva_attacks = all_attacks[N_STANDARD:]
        benign      = [r for r in records if r["true_label"] == 0]

    print(f"  Standard attacks : {len(std_attacks)}  (expected {N_STANDARD})")
    print(f"  Evasion attacks  : {len(eva_attacks)}  (expected {N_EVASION})")
    print(f"  Benign           : {len(benign)}  (expected {N_BENIGN})")

    # ── Confusion values ──────────────────────────────────────────────────────
    std_tp = sum(1 for r in std_attacks if _is_detected(r.get("action", "")))
    std_fn = len(std_attacks) - std_tp
    std_fp = sum(1 for r in benign     if _is_detected(r.get("action", "")))
    std_tn = len(benign) - std_fp

    eva_tp = sum(1 for r in eva_attacks if _is_detected(r.get("action", "")))
    eva_fn = len(eva_attacks) - eva_tp

    print(f"\n  Standard : TP={std_tp}, FN={std_fn}, FP={std_fp}, TN={std_tn}")
    print(f"  Evasion  : TP={eva_tp}, FN={eva_fn}")
    print(f"\n  Expected standard : TP=88, FN=43, FP=67, TN=486")
    print(f"  Expected evasion  : TP=6,  FN=1")

    # ── Layer attribution ─────────────────────────────────────────────────────
    def _layer_counts(subset: list) -> Counter:
        c = Counter()
        for r in subset:
            if _is_detected(r.get("action", "")):
                layer = (r.get("blocking_layer")
                         or "Meta Aggregator - Combined Risk")
                c[layer] += 1
        return c

    std_layers = _layer_counts(std_attacks)
    eva_layers = _layer_counts(eva_attacks)

    print(f"\n  Standard layers : {dict(std_layers)}")
    print(f"  Evasion layers  : {dict(eva_layers)}")
    print(f"\n  Expected standard : L1=57, L2=27, L3=4")
    print(f"  Expected evasion  : L1=1,  L2=4,  L3=1")

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(6.5, 5.5))

    ax1 = fig.add_subplot(2, 2, 1)
    _plot_cm_standard(
        ax1,
        np.array([[std_tp, std_fn], [std_fp, std_tn]]),
        f"Standard (n={len(std_attacks)} attacks, n={len(benign)} benign)",
    )

    ax2 = fig.add_subplot(2, 2, 2)
    _plot_cm_evasion(
        ax2, eva_tp, eva_fn,
        f"Evasion Probes (n={len(eva_attacks)})",
    )

    ax3 = fig.add_subplot(2, 2, 3)
    _plot_attribution(
        ax3, std_layers, std_tp,
        "Standard: Layer Attribution",
    )

    ax4 = fig.add_subplot(2, 2, 4)
    _plot_attribution(
        ax4, eva_layers, eva_tp,
        "Evasion: Layer Attribution",
    )

    plt.tight_layout(h_pad=1.8, w_pad=1.0)
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"fig3_confusion_attribution.{ext}")
        fig.savefig(path)
    plt.close()
    print(f"\n  Saved : fig3_confusion_attribution.pdf / .png")


# ─────────────────────────────────────────────────────────────────────────────
# Subplot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_cm_standard(ax, cm: np.ndarray, title: str) -> None:
    """
    2×2 confusion matrix with count + row-percentage.
    Row 0 = Real Attack  [TP, FN]
    Row 1 = Real Benign  [FP, TN]
    """
    ax.imshow(cm, cmap=_make_cmap(), aspect='auto',
              vmin=0, vmax=max(int(cm.max()), 1))

    cell_labels = [['TP', 'FN'], ['FP', 'TN']]
    row_totals  = [cm[i, :].sum() for i in range(2)]
    for i in range(2):
        for j in range(2):
            val    = cm[i, j]
            pct    = (val / row_totals[i] * 100) if row_totals[i] > 0 else 0
            colour = 'white' if val > cm.max() * 0.55 else 'black'
            ax.text(j, i,
                    f"{cell_labels[i][j]}\n{int(val)}\n({pct:.1f}%)",
                    ha='center', va='center',
                    fontsize=9, fontweight='bold', color=colour)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred Attack', 'Pred Benign'], fontsize=7)
    ax.set_yticklabels(['Real Attack', 'Real Benign'], fontsize=7)
    ax.set_title(title, fontsize=8, fontweight='bold')


def _plot_cm_evasion(ax, tp: int, fn: int, title: str) -> None:
    """
    1×2 matrix for evasion set (attack queries only).
    No benign row — FPR is evaluated on standard corpus.
    """
    total  = tp + fn
    cm     = np.array([[tp, fn]])
    ax.imshow(cm, cmap=_make_cmap(), aspect='auto',
              vmin=0, vmax=max(total, 1))

    for j, (label, val) in enumerate(zip(['TP', 'FN'], [tp, fn])):
        pct    = (val / total * 100) if total > 0 else 0
        colour = 'white' if val > total * 0.55 else 'black'
        ax.text(j, 0,
                f"{label}\n{val}\n({pct:.0f}%)",
                ha='center', va='center',
                fontsize=12, fontweight='bold', color=colour)

    ax.set_xticks([0, 1])
    ax.set_yticks([0])
    ax.set_xticklabels(['Pred Attack', 'Pred Benign'], fontsize=7)
    ax.set_yticklabels(['Real Attack'], fontsize=7)
    ax.text(
        0.5, -0.36,
        'FPR evaluated on standard 553-sample benign corpus',
        ha='center', va='top', transform=ax.transAxes,
        fontsize=6.5, style='italic', color='#555555',
    )
    ax.set_title(title, fontsize=8, fontweight='bold')


def _plot_attribution(ax, layer_counts: Counter,
                      total_tp: int, title: str) -> None:
    """
    Pie chart showing which layer caught each attack.
    Uses a legend (no wedge labels) to prevent label overlap on small slices.
    """
    if total_tp == 0 or not layer_counts:
        ax.text(0.5, 0.5, "No detections",
                ha='center', va='center', fontsize=9,
                transform=ax.transAxes)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.axis('off')
        return

    name_map = {
        "Layer 1 - Anomaly Detection":     "L1: Anomaly",
        "Layer 2 - Intent Classifier":     "L2: Intent",
        "Layer 3 - Behavioral Monitor":    "L3: Semantic",
        "Meta Aggregator - Combined Risk": "Meta-Agg",
    }
    color_map = {
        "L1: Anomaly":  '#1565C0',
        "L2: Intent":   '#E65100',
        "L3: Semantic": '#2E7D32',
        "Meta-Agg":     '#6A1B9A',
    }

    legend_labels = []
    sizes         = []
    colours       = []

    for layer, count in layer_counts.most_common():
        short = name_map.get(layer, layer[:14])
        pct   = count / total_tp * 100
        legend_labels.append(f"{short}: {count} ({pct:.1f}%)")
        sizes.append(count)
        colours.append(color_map.get(short, '#757575'))

    wedges, _ = ax.pie(
        sizes,
        colors=colours,
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
        pctdistance=0.75,
    )
    # Add percentage text inside each wedge manually (clean look)
    for wedge, count in zip(wedges, sizes):
        pct = count / total_tp * 100
        if pct >= 5:   # only label wedges >= 5% to avoid clutter
            angle   = (wedge.theta2 + wedge.theta1) / 2
            x = 0.55 * np.cos(np.deg2rad(angle))
            y = 0.55 * np.sin(np.deg2rad(angle))
            ax.text(x, y, f"{pct:.0f}%",
                    ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white')

    ax.legend(
        wedges, legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.28),
        ncol=1,
        fontsize=6.5,
        frameon=True,
        framealpha=0.9,
        edgecolor='#cccccc',
    )
    ax.set_title(title, fontsize=9, fontweight='bold')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Per-layer latency bar chart  (Section VII-F)
# ─────────────────────────────────────────────────────────────────────────────

def make_fig4_latency(records: list) -> None:
    """
    Horizontal grouped bar chart: mean per-layer inference latency.
    Numbers match Table III from the paper.
    """
    # Latency data from eval_report.json / latency_breakdown.py
    layers    = ['L1: Anomaly\nDetection', 'L2: Intent\nClassifier',
                 'L3: Semantic\nMonitor', 'Meta\nAggregator']
    mean_ms   = [48.2,  312.4, 1870.6,  12.3]   # ms — from latency_breakdown.py
    std_ms    = [ 6.1,   22.8,   95.3,   1.4]

    colours = ['#1565C0', '#E65100', '#2E7D32', '#6A1B9A']

    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    bars = ax.barh(layers, mean_ms, xerr=std_ms,
                   color=colours, alpha=0.85,
                   error_kw={'elinewidth': 1.0, 'capsize': 3, 'ecolor': '#333333'},
                   edgecolor='white', linewidth=0.5)

    for bar, ms in zip(bars, mean_ms):
        ax.text(ms + max(std_ms) * 0.1 + 8, bar.get_y() + bar.get_height() / 2,
                f'{ms:.0f} ms', va='center', fontsize=7)

    ax.set_xlabel('Mean Inference Time (ms)', fontsize=8)
    ax.set_title('Fig. 4 — Per-Component Inference Latency', fontsize=9, fontweight='bold')
    ax.set_xlim([0, max(mean_ms) * 1.30])
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.2, linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT_DIR, f'fig4_latency.{ext}'))
    plt.close()
    print(f'  Saved : fig4_latency.pdf / .png')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Generating IEEE Paper Figures")
    print("  Fig. 2 — ROC + PR Curves")
    print("  Fig. 3 — Confusion Matrices + Layer Attribution")
    print("  Fig. 4 — Per-Component Latency")
    print("=" * 55)

    records = load_data()

    print("\n-- Figure 2: ROC + PR Curves --")
    make_fig2_roc_pr(records)

    print("\n-- Figure 3: Confusion Matrices + Layer Attribution --")
    make_fig3_confusion_attribution(records)

    print("\n-- Figure 4: Per-Layer Latency --")
    make_fig4_latency(records)

    print("\n" + "=" * 55)
    print("  Done. Upload to Overleaf:")
    print("    figures/fig2_roc_pr.png")
    print("    figures/fig3_confusion_attribution.png")
    print("    figures/fig4_latency.png")
    print("=" * 55)