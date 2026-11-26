"""Plot seed-42/43 CSE policy trade-offs after threshold recalibration."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
SEEDS = (42, 43)
NAMES = ("no_adaptation", "threshold_recalibration", "feature_psi", "periodic_5_chunks", "label_error")
LABELS = {"no_adaptation": "No adaptation", "threshold_recalibration": "Threshold only",
          "feature_psi": "PSI refit", "periodic_5_chunks": "Periodic refit", "label_error": "Labeled-error refit"}
COLORS = {"no_adaptation": "#64748b", "threshold_recalibration": "#d97706",
          "feature_psi": "#0284c7", "periodic_5_chunks": "#7c3aed", "label_error": "#059669"}
MARKERS = {42: "o", 43: "s"}

fig, ax = plt.subplots(figsize=(9.1, 5.7), constrained_layout=True)
for name in NAMES:
    points = []
    for seed in SEEDS:
        path = ROOT / "results" / f"cse_cic_ids2018_adaptation_all10_seed{seed}_threshold.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        summary = result["summaries"][name]
        counts = summary["global_confusion_counts"]
        fpr = counts["fp"] / max(1, counts["fp"] + counts["tn"])
        macro_f1 = summary["global_macro_f1_from_confusion"]
        points.append((fpr, macro_f1))
        ax.scatter(fpr, macro_f1, s=85, marker=MARKERS[seed], color=COLORS[name],
                   edgecolor="white", linewidth=0.8, zorder=3,
                   label=f"{LABELS[name]} (seed {seed})" if name == NAMES[0] else None)
    ax.plot([p[0] for p in points], [p[1] for p in points], color=COLORS[name],
            linewidth=1.2, alpha=0.65, zorder=2)
    dx, dy = {"no_adaptation": (8, 8), "threshold_recalibration": (8, -24),
              "feature_psi": (8, 8), "periodic_5_chunks": (8, -20), "label_error": (8, 7)}[name]
    ax.annotate(LABELS[name], points[0], xytext=(dx, dy), textcoords="offset points", fontsize=8.5)
ax.set_xscale("log")
ax.set_xlabel("Pooled false-positive rate (log scale)")
ax.set_ylabel("Pooled macro F1")
ax.set_title("CSE-CIC-IDS2018: policy quality versus false alarms")
ax.grid(True, which="both", alpha=0.25)
ax.set_ylim(0.48, 1.0)
ax.legend(handles=[plt.Line2D([0], [0], marker=MARKERS[s], color="none",
           markerfacecolor="#475569", markeredgecolor="white", markersize=8,
           label=f"Seed {s}") for s in SEEDS], loc="lower right", frameon=True)
ax.text(0.02, 0.02, "40k target chunks; 15.18M rows per seed. Threshold policy uses prior-chunk labels.",
        transform=ax.transAxes, fontsize=8, color="#475569")
OUTPUT = ROOT / "results" / "cse_threshold_recalibration_tradeoff_seeds42_43.png"
fig.savefig(OUTPUT, dpi=180)
print(OUTPUT)
