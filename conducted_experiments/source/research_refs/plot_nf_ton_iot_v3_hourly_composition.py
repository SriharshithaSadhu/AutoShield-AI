"""Plot hourly benign/attack volume and scheduled attack-family composition."""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "nf_ton_iot_v3_hourly_label_counts.csv"
OUTPUT = ROOT / "results" / "nf_ton_iot_v3_hourly_composition.png"

frame = pd.read_csv(SOURCE)
frame["hour_utc"] = pd.to_datetime(frame["hour_utc"], utc=True)
start, end = frame["hour_utc"].min(), frame["hour_utc"].max()
hours = pd.date_range(start, end, freq="h", tz="UTC")
binary = frame.pivot_table(index="hour_utc", columns="binary_label", values="rows", aggfunc="sum", fill_value=0)
binary = binary.reindex(hours, fill_value=0).rename(columns={0: "Benign", 1: "Attack"})
family = frame.loc[frame["binary_label"] == 1].pivot_table(
    index="hour_utc", columns="attack", values="rows", aggfunc="sum", fill_value=0
).reindex(hours, fill_value=0)

plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.titlesize": 15})
fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1.3]}, constrained_layout=True)
axes[0].plot(binary.index, binary["Benign"], color="#276FBF", linewidth=1.5, label="Benign")
axes[0].plot(binary.index, binary["Attack"], color="#D1495B", linewidth=1.5, label="Attack")
axes[0].set_yscale("log")
axes[0].set_ylabel("Completed flows per UTC hour (log scale)")
axes[0].set_title("Binary traffic volume by flow completion hour")
axes[0].grid(True, which="both", axis="y", alpha=0.25)
axes[0].legend(ncol=2, frameon=False, loc="upper left")

colors = plt.get_cmap("tab10").colors
axes[1].stackplot(family.index, [family[name].to_numpy() for name in family.columns],
                  labels=family.columns, colors=colors[:len(family.columns)], alpha=0.9)
axes[1].set_ylabel("Attack-labeled flows per UTC hour")
axes[1].set_title("Attack-family campaign composition (scheduled testbed traffic)")
axes[1].grid(True, axis="y", alpha=0.25)
axes[1].legend(ncol=5, frameon=False, loc="upper left")
axes[1].xaxis.set_major_locator(mdates.DayLocator())
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d", tz=mdates.UTC))
axes[1].set_xlabel("Flow completion time (UTC)")
fig.suptitle("NF-ToN-IoT-v3 chronological composition — descriptive, not production drift")
fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
print(OUTPUT)
