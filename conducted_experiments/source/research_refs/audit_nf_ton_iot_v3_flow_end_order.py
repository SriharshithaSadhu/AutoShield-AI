"""Check whether full-flow rows can be replayed causally at flow end time."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data" / "nf_ton_iot_v3"
SOURCE = ROOT / "NF-ToN-IoT-v3.csv"
OUT = Path(__file__).resolve().parents[1] / "results"
CHUNK = 1_000_000

daily: dict[tuple[str, int, str], int] = defaultdict(int)
invalid_or_missing_end = rows = inversions = negative_intervals = 0
previous_end = None
minimum = maximum = None
maximum_duration_ms = None

for batch_no, frame in enumerate(
    pd.read_csv(
        SOURCE,
        usecols=["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "Label", "Attack"],
        chunksize=CHUNK,
        low_memory=False,
    ),
    start=1,
):
    starts = pd.to_numeric(frame["FLOW_START_MILLISECONDS"], errors="coerce")
    ends = pd.to_numeric(frame["FLOW_END_MILLISECONDS"], errors="coerce")
    labels = pd.to_numeric(frame["Label"], errors="coerce")
    valid = ends.notna() & labels.isin([0, 1])
    invalid_or_missing_end += int((~ends.notna()).sum() + (ends.notna() & ~labels.isin([0, 1])).sum())
    good_end = ends[valid].astype("int64")
    if not good_end.empty:
        minimum = int(good_end.min()) if minimum is None else min(minimum, int(good_end.min()))
        maximum = int(good_end.max()) if maximum is None else max(maximum, int(good_end.max()))
        if previous_end is not None and int(good_end.iloc[0]) < previous_end:
            inversions += 1
        inversions += int((good_end.diff().dropna() < 0).sum())
        previous_end = int(good_end.iloc[-1])
        dates = pd.to_datetime(good_end, unit="ms", utc=True).dt.strftime("%Y-%m-%d")
        good = frame.loc[good_end.index, ["Attack"]].copy()
        good["date_utc"] = dates.to_numpy()
        good["binary_label"] = labels.loc[good_end.index].astype("int8").to_numpy()
        good["attack"] = good["Attack"].fillna("<MISSING>").astype(str).str.strip()
        for key, count in good.groupby(["date_utc", "binary_label", "attack"], sort=False).size().items():
            daily[(key[0], int(key[1]), key[2])] += int(count)
    negative_intervals += int(((ends - starts) < 0).sum())
    batch_max_duration = int((ends - starts).max())
    maximum_duration_ms = batch_max_duration if maximum_duration_ms is None else max(maximum_duration_ms, batch_max_duration)
    rows += len(frame)
    print(f"rows_scanned={rows:,} end_time_inversions={inversions:,}", flush=True)

def utc_ms(value: int | None) -> str | None:
    return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat() if value is not None else None

daily_rows = [
    {"date_utc": date, "binary_label": label, "attack": attack, "rows": count}
    for (date, label, attack), count in sorted(daily.items())
]
pd.DataFrame(daily_rows).to_csv(OUT / "nf_ton_iot_v3_daily_flow_end_counts.csv", index=False)
summary = {
    "rows": rows,
    "invalid_or_missing_end_or_label": invalid_or_missing_end,
    "flow_end_time_inversions_in_publisher_file_order": inversions,
    "negative_end_minus_start_rows": negative_intervals,
    "maximum_flow_duration_ms": maximum_duration_ms,
    "min_flow_end_utc": utc_ms(minimum),
    "max_flow_end_utc": utc_ms(maximum),
    "daily_flow_end_counts_csv": "results/nf_ton_iot_v3_daily_flow_end_counts.csv",
    "causal_scoring_rule": "Use full flow-final features only when the corresponding FLOW_END_MILLISECONDS has arrived; sort by that timestamp before replay if publisher row order is not end-time order.",
}
path = OUT / "nf_ton_iot_v3_flow_end_order_audit.json"
path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
