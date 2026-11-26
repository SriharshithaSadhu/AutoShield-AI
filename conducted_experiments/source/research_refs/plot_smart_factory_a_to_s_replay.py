"""Plot recorded chunk-level metrics from the smart-factory A-to-S replay."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "SMART_FACTORY_A_TO_S_REPLAY.json"
OUTPUT = ROOT / "results" / "smart_factory_a_to_s_chunk_metrics.png"
ARMS = ["no_update", "periodic", "fpr_recall_gate", "threshold_only"]
LABELS = {
    "no_update": "Frozen model",
    "periodic": "Periodic refit",
    "fpr_recall_gate": "FPR + recall gate",
    "threshold_only": "Threshold-only",
}


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    colors = dict(zip(ARMS, plt.get_cmap("tab10").colors[:len(ARMS)]))
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.05, 1]})
    fpr_ax, recall_ax = axes
    boundary = 71.5
    for ax in axes:
        ax.axvspan(boundary, 108.5, color="#eef5fb", zorder=0)
        ax.axvline(boundary, color="#555555", linewidth=1.0, linestyle="--", zorder=1)
        ax.grid(True, which="major", axis="both", alpha=0.22, linewidth=0.65)

    for arm in ARMS:
        seed_series = []
        for run in result["seeds"]:
            chunks = run["chunks"]
            x = np.array([c["chunk"] for c in chunks], dtype=int)
            fpr = np.array([c["arms"][arm]["fpr"] * 100 for c in chunks])
            recall = np.array([c["arms"][arm]["recall"] * 100 for c in chunks])
            seed_series.append((x, fpr, recall))
            fpr_ax.plot(x, fpr, color=colors[arm], alpha=0.23, linewidth=0.75)
            recall_ax.plot(x, recall, color=colors[arm], alpha=0.23, linewidth=0.75)
        fpr_mean = np.mean([v[1] for v in seed_series], axis=0)
        recall_mean = np.mean([v[2] for v in seed_series], axis=0)
        fpr_ax.plot(seed_series[0][0], fpr_mean, color=colors[arm], linewidth=2.0,
                    label=LABELS[arm])
        recall_ax.plot(seed_series[0][0], recall_mean, color=colors[arm], linewidth=2.0)

    fpr_ax.axhline(0.1, color="#b22222", linewidth=1.3, linestyle=(0, (5, 3)),
                   label="0.1% FPR cap")
    fpr_ax.set_yscale("symlog", linthresh=0.01, linscale=0.8)
    fpr_ax.set_ylabel("False-positive rate (%)")
    recall_ax.set_ylabel("Attack recall (%)")
    recall_ax.set_xlabel("Chronological 10,000-flow chunk")
    recall_ax.set_ylim(-2, 102)
    fpr_ax.set_xlim(1, 108)
    fpr_ax.legend(loc="upper left", ncol=2, frameon=True, fontsize=9)

    # Mark the gate's candidate-validation observations. These are decision-time
    # validation points, not the incumbent deployment scores plotted as lines.
    seen = set()
    for run in result["seeds"]:
        for event in run["promotion_events"]["fpr_recall_gate"]:
            key = (event["phase"], event["chunk"])
            if key in seen:
                continue
            seen.add(key)
            val = event["validation"]
            fpr_ax.scatter(event["chunk"], val["fpr"] * 100, marker="^", s=62,
                           facecolor="white", edgecolor=colors["fpr_recall_gate"],
                           linewidth=1.4, zorder=5)
            recall_ax.scatter(event["chunk"], val["recall"] * 100, marker="^", s=62,
                              facecolor="white", edgecolor=colors["fpr_recall_gate"],
                              linewidth=1.4, zorder=5)
    recall_ax.text(0.99, 0.96, "Shaded area: later November S campaign",
                   transform=recall_ax.transAxes, ha="right", va="top", fontsize=9,
                   color="#334455")
    fig.suptitle("Causal adaptation replay: false alarms and attack recall over time",
                 fontsize=14, y=0.99)
    fig.text(0.5, 0.01,
             "Thin lines: individual model random-state settings; thick lines: their mean. White triangles: accepted candidate validation chunks; deployment outcomes occur afterward.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.965))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
