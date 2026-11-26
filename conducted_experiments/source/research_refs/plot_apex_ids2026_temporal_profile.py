"""Plot daily volume and label composition for the audited APEX HC subset."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "apex_ids2026" / "temporal_audit.json"
OUT = ROOT / "results" / "APEX_IDS2026_TEMPORAL_PROFILE.png"


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    frame = pd.DataFrame.from_dict(audit["days"], orient="index")
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    frame["total"] = frame["Attack_Verified"] + frame["Benign_Verified"]
    frame["attack_share"] = frame["Attack_Verified"] / frame["total"] * 100
    overall = audit["label_counts"]["Attack_Verified"] / audit["parquet_rows"] * 100

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.facecolor": "#f7f9fc",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#d5dce5",
        "axes.labelcolor": "#263449",
        "text.color": "#182335",
        "xtick.color": "#536174",
        "ytick.color": "#536174",
    })
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13.5, 8.3), sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1]}
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.85, hspace=0.28)
    dates = frame.index
    ax1.bar(dates, frame["Benign_Verified"], width=0.82, color="#48a9a6",
            label="Benign_Verified", linewidth=0)
    ax1.bar(dates, frame["Attack_Verified"], width=0.82,
            bottom=frame["Benign_Verified"], color="#e28a46",
            label="Attack_Verified", linewidth=0)
    ax1.set_ylabel("Flows per UTC day")
    ax1.set_title("Daily volume in the high-confidence two-label subset", loc="left", weight="bold")
    ax1.legend(frameon=False, ncol=2, loc="upper right")
    ax1.grid(axis="y", color="#e8edf3", linewidth=0.8)
    ax1.set_axisbelow(True)

    ax2.plot(dates, frame["attack_share"], color="#c5652c", linewidth=2.2,
             marker="o", markersize=3.5, label="Attack_Verified share")
    ax2.axhline(overall, color="#44546a", linestyle=(0, (4, 3)), linewidth=1.4,
                label=f"Overall subset share ({overall:.1f}%)")
    ax2.fill_between(dates, frame["attack_share"], overall, color="#e28a46", alpha=0.12)
    ax2.set_ylabel("Attack share (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("Daily label composition (selected subset; not operational prevalence)", loc="left", weight="bold")
    ax2.grid(axis="y", color="#e8edf3", linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, ncol=2, loc="upper right")
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.set_xlabel("Flow-start date (UTC, publisher-defined)")

    fig.suptitle("APEX-IDS2026 temporal profile · 69.1M verified-label flows",
                 x=0.04, y=0.975, ha="left", fontsize=17, weight="bold", color="#152238")
    fig.text(0.04, 0.925,
             "One ISP perimeter · 44 days · Attack_Verified vs Benign_Verified · event-time profile only",
             ha="left", fontsize=10, color="#526174")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
