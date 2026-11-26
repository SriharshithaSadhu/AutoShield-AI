"""Plot the released WUSTL-IIoT-2021 labels by flow end time (audit only)."""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
CSV = ROOT / "data" / "wustl_iiot" / "wustl_iiot_2021.csv"
OUT = ROOT / "results" / "wustl_temporal_label_distribution.png"


def main() -> None:
    parts = []
    for frame in pd.read_csv(CSV, usecols=["LastTime", "Traffic"], chunksize=200_000):
        frame["LastTime"] = pd.to_datetime(frame["LastTime"], errors="coerce")
        frame = frame.dropna(subset=["LastTime"])
        frame["bin"] = frame["LastTime"].dt.floor("10min")
        parts.append(frame.groupby(["bin", "Traffic"]).size())
    counts = pd.concat(parts).groupby(level=[0, 1]).sum().unstack(fill_value=0)
    order = [x for x in ["normal", "DoS", "Reconn", "CommInj", "Backdoor"] if x in counts]
    counts = counts[order]
    colors = {"normal": "#577590", "DoS": "#f94144", "Reconn": "#f9c74f",
              "CommInj": "#90be6d", "Backdoor": "#9b5de5"}

    fig, ax = plt.subplots(figsize=(12, 5.6), constrained_layout=True)
    counts.plot(kind="bar", stacked=True, ax=ax,
                color=[colors[x] for x in order], width=0.92, edgecolor="none")
    ax.set_title("WUSTL-IIoT-2021 released CSV: labels by 10-minute flow-end time")
    ax.set_xlabel("LastTime (flow completion / availability time)")
    ax.set_ylabel("Rows per 10-minute bin")
    ax.legend(title="Traffic", ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ticks = ax.get_xticks()
    labels = [pd.Timestamp(counts.index[i]).strftime("%H:%M") if i % 3 == 0 else ""
              for i in range(len(counts.index))]
    ax.set_xticks(ticks, labels, rotation=0)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)
    print(f"Saved {OUT}")
    print("Total rows plotted:", int(counts.to_numpy().sum()))


if __name__ == "__main__":
    main()
