"""Plot the label-category pulse around the July 29 low-recall bin."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "apex_ids2026_failure_attribution_20261107.json"
OUT = ROOT / "results" / "APEX_IDS2026_RECONNAISSANCE_PULSE.png"


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    focal = data["target_bins"]["lowest_recall"]
    before = focal["adjoining_30min_summary"]["before"]
    after = focal["adjoining_30min_summary"]["after"]
    target_counts = focal["confusion"]
    target_total = sum(target_counts.values())
    target_recon = (focal["by_outcome"]["false_negative"]["attack_category_top10"].get("reconnaissance", 0)
                    + focal["by_outcome"]["true_positive"]["attack_category_top10"].get("reconnaissance", 0))
    items = [("Previous 30 min\n(6 × 5-min bins)", before["rows"],
              before["attack_category_top10"].get("reconnaissance", 0),
              before["attack_category_top10"].get("benign", 0)),
             ("Low-recall 5-min\nbin", target_total, target_recon,
              focal["by_outcome"]["false_positive"]["attack_category_top10"].get("benign", 0)
              + focal["by_outcome"]["true_negative"]["attack_category_top10"].get("benign", 0)),
             ("Following 30 min\n(6 × 5-min bins)", after["rows"],
              after["attack_category_top10"].get("reconnaissance", 0),
              after["attack_category_top10"].get("benign", 0))]
    labels = [x[0] for x in items]
    recon_pct = np.array([100 * x[2] / x[1] for x in items])
    attack_pct = np.array([100 * (x[1] - x[3]) / x[1] for x in items])
    x = np.arange(3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=False)
    colors = ["#4c78a8", "#e45756", "#4c78a8"]
    ax1.bar(x, recon_pct, color=colors, width=.62)
    ax1.set_ylabel("Publisher-coded reconnaissance share (%)")
    ax1.set_title("T1046/scan category pulse")
    ax1.set_ylim(0, 80)
    ax1.grid(axis="y", alpha=.25)
    for i, (pct, item) in enumerate(zip(recon_pct, items)):
        ax1.text(i, pct + 1.4, f"{pct:.1f}%\n({item[2]:,}/{item[1]:,})", ha="center", fontsize=9)
    ax2.bar(x, attack_pct, color=colors, width=.62)
    ax2.set_ylabel("Publisher binary attack-label share (%)")
    ax2.set_title("Attack-label mix in selected subset")
    ax2.set_ylim(0, 80)
    ax2.grid(axis="y", alpha=.25)
    for i, (pct, item) in enumerate(zip(attack_pct, items)):
        attack_n = item[1] - item[3]
        ax2.text(i, pct + 1.4, f"{pct:.1f}%\n({attack_n:,}/{item[1]:,})", ha="center", fontsize=9)
    for ax in (ax1, ax2):
        ax.set_xticks(x, labels)
        ax.set_xlim(-.6, 2.6)
    fig.suptitle("Observed event-time composition pulse around the Jul 29 recall trough",
                 fontsize=12, fontweight="bold", y=.98)
    fig.tight_layout(rect=[0, .11, 1, .91])
    fig.text(.5, .025, "High-confidence selected subset only; not operational attack prevalence or proof of causal drift.",
             ha="center", fontsize=9)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
