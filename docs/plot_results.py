"""Regenerate portfolio graphs from saved, measured results (no model fitting)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "results/graphs"


def main():
    DEST.mkdir(exist_ok=True)
    metrics = json.loads((ROOT / "results/metrics.json").read_text())["methods"]
    bench = json.loads((ROOT / "results/resource_benchmark.json").read_text())[
        "records"
    ]
    labels = ["Flow-only", "Hybrid"]
    colors = ["#355C7D", "#087F8C"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    for ax, key, title in zip(
        axes, ["recall", "fpr"], ["Attack recall (%)", "False-positive rate (%)"]
    ):
        values = [100 * row[key] for row in metrics]
        bars = ax.bar(labels, values, color=colors, width=0.55)
        ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=6)
        ax.set_ylabel(title)
        ax.set_ylim(0, max(values) * 1.2)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    axes[1].axhline(0.1, color="#B64B39", ls="--", lw=1.5, label="FPR target: 0.1%")
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "Detection quality and false-alarm tradeoff", weight="bold", fontsize=14
    )
    fig.supxlabel(
        "9,160,253 chronological ToN development records Â· frozen seed 42", fontsize=10
    )
    fig.savefig(DEST / "overview_detection_tradeoff.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    sets = [
        [r["service"]["p95_ms"] for r in bench],
        [r["model_bytes"] / 1024 for r in bench],
    ]
    for ax, values, title, precision in zip(
        axes,
        sets,
        ["Single-flow service p95 (ms)", "Serialized model size (KiB)"],
        [3, 1],
    ):
        bars = ax.bar(labels, values, color=colors, width=0.55)
        ax.bar_label(bars, labels=[f"{v:.{precision}f}" for v in values], padding=6)
        ax.set_ylabel(title)
        ax.set_ylim(0, max(values) * 1.22)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle("Measured desktop inference cost", weight="bold", fontsize=14)
    fig.supxlabel(
        "5,000 real records after 1,000 warmup Â· excludes capture, transport and queue wait",
        fontsize=10,
    )
    fig.savefig(DEST / "overview_resource_cost.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
