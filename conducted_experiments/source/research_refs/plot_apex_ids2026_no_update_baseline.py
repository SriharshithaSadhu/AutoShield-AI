"""Render the locked APEX no-update baseline's day and five-minute tails."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "apex_ids2026_no_update_seed43.json"
OUT = ROOT / "results" / "APEX_IDS2026_NO_UPDATE_TEMPORAL_RESULTS.png"


def main() -> None:
    result = json.loads(SOURCE.read_text(encoding="utf-8"))
    daily = result["evaluation"]["daily"]
    days = pd.to_datetime(list(daily.keys()), utc=True)
    daily_fpr = np.array([daily[d]["fpr"] for d in daily]) * 100
    daily_recall = np.array([daily[d]["recall"] for d in daily]) * 100
    all_bins = result["evaluation"]["five_minute_bins"]
    bins = [b for b in all_bins if b["eligible_for_worst_bin"]]
    times = pd.to_datetime([b["bin_start_utc"] for b in bins], utc=True)
    fpr = np.array([b["fpr"] for b in bins]) * 100
    eligible = fpr > 0.1

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "figure.dpi": 150, "savefig.dpi": 200})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8.2), constrained_layout=True)
    ax1.plot(days, daily_fpr, color="#b23a48", marker="o", linewidth=1.8,
             label="Daily FPR")
    ax1.axhline(0.1, color="#8b1e2d", linestyle="--", linewidth=1.3,
                label="Development cap: 0.1%")
    ax1.set_ylabel("Benign false-positive rate (%)")
    ax1.set_ylim(bottom=0)
    ax1.grid(axis="y", alpha=0.25)
    ax1b = ax1.twinx()
    ax1b.plot(days, daily_recall, color="#246a73", marker="s", linewidth=1.5,
              label="Daily recall")
    ax1b.set_ylabel("Attack recall (%)")
    ax1b.set_ylim(85, 100)
    ax1b.xaxis.set_major_formatter(mdates.DateFormatter("%b %d", tz=mdates.UTC))
    ax1.set_title("Frozen threshold on the final 11-day interval")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", ncols=3, frameon=False)

    ax2.scatter(times[~eligible], fpr[~eligible], s=9, color="#4c78a8", alpha=0.58,
                label="Supported bin at or below cap")
    ax2.scatter(times[eligible], fpr[eligible], s=11, color="#e45756", alpha=0.68,
                label="Supported bin above cap")
    ax2.axhline(0.1, color="#8b1e2d", linestyle="--", linewidth=1.3,
                label="0.1% cap")
    worst = max(bins, key=lambda b: b["fpr"])
    worst_t = pd.to_datetime(worst["bin_start_utc"], utc=True)
    ax2.annotate(f"Worst: {worst['fpr'] * 100:.2f}%\n{worst_t:%b %d %H:%M} UTC",
                 xy=(worst_t, worst["fpr"] * 100), xytext=(-100, -42),
                 textcoords="offset points", arrowprops={"arrowstyle": "->", "lw": 0.9},
                 fontsize=9)
    ax2.set_ylabel("Five-minute benign FPR (%)")
    ax2.set_xlabel("UTC event-time bin (flow start + duration)")
    ax2.set_ylim(bottom=0)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d", tz=mdates.UTC))
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="upper left", ncols=3, frameon=False)
    fig.suptitle("APEX-IDS2026 CPU reference — curated verified-flow endpoint",
                 fontsize=13, fontweight="bold")
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
